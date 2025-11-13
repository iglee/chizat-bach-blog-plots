import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import trange
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# -----------------------
# Precision / RNG
# -----------------------
SEED = 0
torch.set_default_dtype(torch.float64)
torch.manual_seed(SEED)
np.random.seed(SEED)


# -----------------------
# Model: Linear layers, but same behavior as Julia
# fc1: (d -> m) with bias; fc2: (m -> 1) no bias; divide output by m
# Row-normalize [bias, w1, w2] per neuron (just like Julia's W[:, 1:d] normalization with bias inside)
# -----------------------
class TwoNetLinear(nn.Module):
    def __init__(self, d_in: int, m: int):
        super().__init__()
        self.m = m
        self.fc1 = nn.Linear(d_in, m, bias=True)   # corresponds to W[:, :d] and the bias column
        self.fc2 = nn.Linear(m, 1, bias=False)     # corresponds to W[:, end]
        self._init_like_julia()

    def _init_like_julia(self):
        with torch.no_grad():
            # Normalize each neuron's parameter vector [bias, w_row...] to unit norm
            W = self.fc1.weight.detach().clone()   # (m, d_in)
            b = self.fc1.bias.detach().clone()     # (m,)
            m, d = W.shape
            concat = torch.cat([b.view(m, 1), W], dim=1)  # (m, d+1)
            norms = concat.pow(2).sum(dim=1, keepdim=True).sqrt().clamp_min(1e-12)
            concat = concat / norms
            self.fc1.bias.copy_(concat[:, 0])
            self.fc1.weight.copy_(concat[:, 1:])

            # Output weights: first half +1, second half -1
            w2 = torch.ones(self.m, dtype=torch.float64)
            w2[self.m // 2:] = -1.0
            self.fc2.weight.copy_(w2.view(1, -1))

    def forward(self, x):  # x: (N, d_in)
        h = F.relu(self.fc1(x))          # (N, m)
        y = self.fc2(h).squeeze(-1)      # (N,)
        return y / self.m


# -----------------------
# Exact manual training loop (same math/signs as Julia)
# - Labels Y in {-1, +1}
# - Loss: softplus(-Y * f)
# - Schedule: stepsize / sqrt(iter+1) with 1-based iter => (it + 2) here
# - L2 applied to fc1.weight, fc1.bias, fc2.weight (i.e., all of W including bias)
# Returns Julia-style snapshots Ws: (m, 4, niter) = [bias, w1, w2, w_out]
# -----------------------
def train_twonet_linear(
    model: TwoNetLinear,
    Xtr: np.ndarray, Ytr: np.ndarray,
    Xte: np.ndarray, Yte: np.ndarray,
    stepsize: float, niter: int, weight_decay: float
):
    device = torch.device("cpu")
    model.to(device)

    Xtr_t = torch.from_numpy(Xtr).to(device)            # (ntr, d)
    Ytr_t = torch.from_numpy(Ytr).to(torch.float64)     # (ntr,)
    Xte_t = torch.from_numpy(Xte).to(device)
    Yte_t = torch.from_numpy(Yte).to(torch.float64)

    ntr, d = Xtr_t.shape
    m = model.m

    # logs
    Ws           = np.zeros((m, d + 2, niter), dtype=np.float64)  # (m, 4, T) for d=2
    train_losses = np.zeros(niter)
    test_losses  = np.zeros(niter)
    margins      = np.zeros(niter)
    betas        = np.zeros(niter)
    train_accs   = np.zeros(niter)
    test_accs    = np.zeros(niter)

    for it in trange(niter, desc="Training neural network..."):
        # ------ snapshot BEFORE update (matches Julia) ------
        with torch.no_grad():
            w_in = model.fc1.weight.data.cpu().numpy()            # (m, d)
            b_in = model.fc1.bias.data.cpu().numpy()              # (m,)
            w2   = model.fc2.weight.data.cpu().numpy().reshape(-1)  # (m,)
            W_full = np.column_stack([b_in, w_in, w2])            # (m, d+2)
            Ws[:, :, it] = W_full

        # ------ forward (train) ------
        with torch.no_grad():
            pre_tr = (model.fc1.weight @ Xtr_t.T) + model.fc1.bias.view(-1, 1)  # (m, ntr)
            act_tr = F.relu(pre_tr)                                             # (m, ntr)
            out_tr = (model.fc2.weight.view(-1, 1) * act_tr).sum(dim=0) / m     # (ntr,)
            perf_tr = Ytr_t * out_tr

            # metrics
            train_losses[it] = torch.log1p(torch.exp(-perf_tr)).mean().item()
            margins[it]      = perf_tr.min().item()
            train_accs[it]   = (perf_tr > 0).to(torch.float64).mean().item()

        # ------ gradients (exact Julia math) ------
        # gradR = -Y * σ(-perf) / n
        gradR = -(Ytr_t * torch.sigmoid(-perf_tr)) / ntr  # (ntr,)

        gate = (pre_tr > 0).to(torch.float64)                               # (m, ntr)
        grad_fc1_w = (model.fc2.weight.view(-1, 1) * gate) @ (Xtr_t * gradR.view(-1, 1))  # (m, d)
        grad_fc1_b = (model.fc2.weight.view(-1) * (gate @ gradR))                           # (m,)
        grad_fc2_w = (act_tr @ gradR)                                                      # (m,)

        # L2: add weight_decay * parameter
        with torch.no_grad():
            grad_fc1_w += weight_decay * model.fc1.weight
            grad_fc1_b += weight_decay * model.fc1.bias
            grad_fc2_w += weight_decay * model.fc2.weight.view(-1)

        # ------ step (Julia schedule) ------
        eta = stepsize / math.sqrt(it + 2)  # iter=1 -> sqrt(2) in Julia; here it=0 -> sqrt(2)
        with torch.no_grad():
            model.fc1.weight -= eta * grad_fc1_w
            model.fc1.bias   -= eta * grad_fc1_b
            model.fc2.weight -= eta * grad_fc2_w.view(1, -1)

        # track squared norm / m
        with torch.no_grad():
            sq = model.fc1.weight.pow(2).sum() + model.fc1.bias.pow(2).sum() + model.fc2.weight.pow(2).sum()
            betas[it] = sq.item() / m

        # ------ forward (test) ------
        with torch.no_grad():
            pre_te  = (model.fc1.weight @ Xte_t.T) + model.fc1.bias.view(-1, 1)
            act_te  = F.relu(pre_te)
            out_te  = (model.fc2.weight.view(-1, 1) * act_te).sum(dim=0) / m
            perf_te = Yte_t * out_te
            test_losses[it] = torch.log1p(torch.exp(-perf_te)).mean().item()
            test_accs[it]   = (perf_te > 0).to(torch.float64).mean().item()

    return (
        Ws, train_losses, margins, betas,
        train_accs, test_losses, test_accs
    )


# -----------------------
# Data generation (same geometry as Julia, but WITHOUT bias in X;
# bias is handled by fc1.bias so we keep X as (x1, x2)).
# -----------------------
def cluster_center(p: np.ndarray, k: int):
    p = p.astype(np.int64)
    p1 = (p - 1) % k + 1
    p2 = (p - 1) // k + 1
    Δ = 1.0 / (3 * k - 1)
    x1 = Δ * (1 + 3 * (p1 - 1)) - 0.5
    x2 = Δ * (1 + 3 * (p2 - 1)) - 0.5
    return x1, x2


def make_data(k: int, n: int):
    Δ = 1.0 / (3 * k - 1)
    A = np.ones(k * k)
    A[np.random.permutation(k * k)[: (k * k) // 2]] = -1

    P = np.random.randint(1, k * k + 1, size=n)
    T = 2 * np.pi * np.random.rand(n)
    R = Δ * np.random.rand(n)
    x1c, x2c = cluster_center(P, k)
    # X WITHOUT bias column (we use fc1.bias instead)
    X = np.column_stack([x1c + R * np.cos(T), x2c + R * np.sin(T)])  # (n, 2)
    Y = A[P - 1]

    # 40/60 split
    idx = np.random.permutation(n)
    ntr = int(np.floor(0.4 * n))
    tr_idx, te_idx = idx[:ntr], idx[ntr:]
    Xtr, Ytr = X[tr_idx, :], Y[tr_idx]
    Xte, Yte = X[te_idx, :], Y[te_idx]
    return Xtr, Ytr, Xte, Yte, X, Y


# -----------------------
# Projections & Visualization (same as Julia)
# We reconstruct W = [bias, w1, w2, w_out] from the model snapshots.
# -----------------------
def activation_log(Ws_snapshots: np.ndarray, X_aug: np.ndarray):
    """
    μ_i(t) = (1/n) Σ_x ReLU(w_i(t)·x_aug) x_aug
    Ws_snapshots: (m, 4, T) with cols [b, w1, w2, w_out]
    X_aug: (n, 3) = [1, x1, x2]
    Returns: (m, 3, T)
    """
    m, _, T = Ws_snapshots.shape
    out = np.zeros((m, 3, T), dtype=np.float64)
    nX = X_aug.shape[0]
    for t in range(T):
        Wt_in = Ws_snapshots[:, :3, t]           # (m,3)
        pre = Wt_in @ X_aug.T                    # (m,nX)
        act = np.maximum(pre, 0.0)
        M = (act @ X_aug) / nX                   # (m,3)
        r = np.sqrt((M ** 2).sum(axis=1, keepdims=True))
        dir_ = M / (r + 1e-8)
        #out[:, :, t] = np.tanh(5.0 * r) * dir_
        out[:, :, t] = np.tanh(10.0 * r) * dir_
    return out


def illustration(k, n, m, stepsize, niter, nframes, resolution):
    # --- data ---
    Xtr, Ytr, Xte, Yte, Xfull, Yfull = make_data(k, n)
    Xtr_aug = np.column_stack([np.ones(Xtr.shape[0]), Xtr])   # (ntr,3)
    Xte_aug = np.column_stack([np.ones(Xte.shape[0]), Xte])   # (nte,3)
    Xfull_aug = np.column_stack([np.ones(Xfull.shape[0]), Xfull])

    # for boundary scatter
    X1 = Xfull[Yfull ==  1.0, :]
    X2 = Xfull[Yfull == -1.0, :]

    # --- model & train (weight_decay=1e-3 like your Julia viz) ---
    model = TwoNetLinear(d_in=2, m=m)
    (
        Ws, train_losses, margins, betas, train_accs, test_losses, test_accs
    ) = train_twonet_linear(
        model, Xtr, Ytr, Xte, Yte,
        stepsize=stepsize, niter=niter, weight_decay=1e-3
    )

    # --- time subsampling (quartic spacing) ---
    a = (niter - 1) / (nframes - 1) ** 4 if nframes > 1 else 0.0
    ts = np.unique((np.floor(a * (np.arange(nframes) ** 4)).astype(int) + 1))
    ts = ts[(ts >= 1) & (ts <= niter)]
    Ws = Ws[:, :, ts - 1]  # (m, 4, T)
    Tframes = Ws.shape[-1]

    # (A) parameter-space projection (data-agnostic)
    Wproj = Ws[:, :3, :] * np.abs(Ws[:, 3:4, :])                 # (m,3,T) scale by |w_out|
    WN    = np.sqrt((Wproj ** 2).sum(axis=1, keepdims=True))     # (m,1,T)
    Wdir  = Wproj / (WN + 1e-12)
    Wlog  = np.tanh(0.5 * WN) * Wdir                             # (m,3,T)

    # (B) activation-space projection (train/test)
    Wlog_act_tr = activation_log(Ws, Xtr_aug)
    Wlog_act_te = activation_log(Ws, Xte_aug)

    # --- plotting loop (2×3 grid) ---
    for kf in trange(Tframes, desc="Plotting images..."):
        fig = plt.figure(figsize=(18, 8))
        gs = fig.add_gridspec(2, 3, height_ratios=[1.2, 0.9], wspace=0.35, hspace=0.35)

        indt = np.arange(0, kf + 1) if kf < 10 else np.arange(kf - 10, kf + 1)
        half = m // 2

        # (1) PARAM: x->w1, y->w2, z->bias
        axP = fig.add_subplot(gs[0, 0], projection="3d")
        for i in range(Wlog.shape[0]):
            axP.plot(Wlog[i, 1, indt], Wlog[i, 2, indt], Wlog[i, 0, indt], color="k", linewidth=0.2)
        axP.plot(Wlog[:half, 1, kf], Wlog[:half, 2, kf], Wlog[:half, 0, kf], "o", color="C3", markersize=2)
        axP.plot(Wlog[half:, 1, kf], Wlog[half:, 2, kf], Wlog[half:, 0, kf], "o", color="C0", markersize=2)
        axP.set_xlim(-1,1); axP.set_ylim(-1,1); axP.set_zlim(-1,1)
        axP.set_title("Neuron trajectories (parameter)")

        # (2) TRAIN activations
        axT = fig.add_subplot(gs[0, 1], projection="3d")
        for i in range(Wlog_act_tr.shape[0]):
            axT.plot(Wlog_act_tr[i, 1, indt], Wlog_act_tr[i, 2, indt], Wlog_act_tr[i, 0, indt], color="k", linewidth=0.2)
        axT.plot(Wlog_act_tr[:half, 1, kf], Wlog_act_tr[:half, 2, kf], Wlog_act_tr[:half, 0, kf], "o", color="C3", markersize=2)
        axT.plot(Wlog_act_tr[half:, 1, kf], Wlog_act_tr[half:, 2, kf], Wlog_act_tr[half:, 0, kf], "o", color="C0", markersize=2)
        axT.set_xlim(-1,1); axT.set_ylim(-1,1); axT.set_zlim(-1,1)
        axT.set_title("Neuron activation trajectories (train)")

        # (3) TEST activations
        axE = fig.add_subplot(gs[0, 2], projection="3d")
        for i in range(Wlog_act_te.shape[0]):
            axE.plot(Wlog_act_te[i, 1, indt], Wlog_act_te[i, 2, indt], Wlog_act_te[i, 0, indt], color="k", linewidth=0.2)
        axE.plot(Wlog_act_te[:half, 1, kf], Wlog_act_te[:half, 2, kf], Wlog_act_te[:half, 0, kf], "o", color="C3", markersize=2)
        axE.plot(Wlog_act_te[half:, 1, kf], Wlog_act_te[half:, 2, kf], Wlog_act_te[half:, 0, kf], "o", color="C0", markersize=2)
        axE.set_xlim(-1,1); axE.set_ylim(-1,1); axE.set_zlim(-1,1)
        axE.set_title("Neuron activation trajectories (test)")

        # (4) Decision boundary from snapshot kf
        axB = fig.add_subplot(gs[1, 0])
        xs = np.arange(-0.8, 0.8 + 1e-12, resolution)
        tab = np.zeros((len(xs), len(xs)))
        Wsnap = Ws[:, :, kf]               # (m, 4)
        W_in = Wsnap[:, :3]                # [bias, w1, w2]
        w_out = Wsnap[:, 3]                # (m,)

        def f_val(x1, x2):
            v = np.array([1.0, x1, x2])
            pre = W_in @ v
            act = np.maximum(pre, 0.0)
            return np.mean(w_out * act)

        for i in range(len(xs)):
            for j in range(len(xs)):
                tab[i, j] = f_val(xs[i], xs[j])

        axB.pcolormesh(xs, xs, np.tanh(tab.T), cmap="coolwarm", shading="gouraud", vmin=-1.0, vmax=1.0)
        axB.contour(xs, xs, np.tanh(tab.T), levels=[0.0], colors="k", linewidths=2)
        axB.plot(X1[:, 0], X1[:, 1], "+k"); axB.plot(X2[:, 0], X2[:, 1], "_k")
        axB.set_aspect("equal"); axB.axis("off")
        axB.set_title("Decision boundary")

        # (5) Loss
        axL = fig.add_subplot(gs[1, 1])
        axL.plot(np.arange(1, len(train_losses) + 1), train_losses, label="Train")
        axL.plot(np.arange(1, len(test_losses) + 1), test_losses, linestyle="--", label="Test")
        axL.axvline((ts[kf]), color="k", linestyle=":", linewidth=1)
        axL.set_xlabel("Iteration"); axL.set_ylabel("Cross Entropy Loss")
        axL.set_title("Loss (train vs test)")
        axL.grid(True); axL.legend(loc="upper right", frameon=False)

        # (6) Accuracy
        axA = fig.add_subplot(gs[1, 2])
        axA.plot(np.arange(1, len(train_accs) + 1), train_accs, label="Train")
        axA.plot(np.arange(1, len(test_accs) + 1), test_accs, linestyle="--", label="Test")
        axA.axvline((ts[kf]), color="k", linestyle=":", linewidth=1)
        axA.set_xlabel("Iteration"); axA.set_ylabel("Accuracy"); axA.set_ylim(0, 1.0)
        axA.set_title("Accuracy (train vs test)")
        axA.grid(True); axA.legend(loc="upper right", frameon=False)

        plt.savefig(f"dynamics_wd_{kf+1}.png", bbox_inches="tight", dpi=300)
        plt.close(fig)


# -----------------------
# Main (demo)
# -----------------------
if __name__ == "__main__":
    # Match your Julia-scale experiment
    k, n, m = 3, 600, 200
    stepsize, niter = 1.0, 20000
    nframes, resolution = 200, 0.02
    illustration(k, n, m, stepsize, niter, nframes, resolution)
