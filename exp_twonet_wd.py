# twonet_exact_replica_fixed.py
import math
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import trange
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

SEED = 0
torch.set_default_dtype(torch.float64)
torch.manual_seed(SEED)
np.random.seed(SEED)

def twonet(Xtr, Ytr, m, stepsize, niter, *, weight_decay=1e-4, Xte=None, Yte=None):
    Xtr_t = torch.from_numpy(Xtr)                       # (ntr, d) with bias in col 0
    Ytr_t = torch.from_numpy(Ytr).to(torch.float64)     # (ntr,)
    if Xte is None or Yte is None:
        Xte_t, Yte_t = Xtr_t, Ytr_t
    else:
        Xte_t = torch.from_numpy(Xte)
        Yte_t = torch.from_numpy(Yte).to(torch.float64)

    ntr, d = Xtr_t.shape
    nte = Xte_t.shape[0]

    # init like Julia
    W = torch.randn(m, d + 1, dtype=torch.float64)
    W[:, :d] = W[:, :d] / (W[:, :d].pow(2).sum(dim=1, keepdim=True).sqrt() + 1e-12)
    W[: m // 2, -1] =  1.0
    W[m // 2 :, -1] = -1.0

    Ws           = torch.zeros(m, d + 1, niter, dtype=torch.float64)
    train_losses = torch.zeros(niter, dtype=torch.float64)
    test_losses  = torch.zeros(niter, dtype=torch.float64)
    margins      = torch.zeros(niter, dtype=torch.float64)
    betas        = torch.zeros(niter, dtype=torch.float64)
    train_accs   = torch.zeros(niter, dtype=torch.float64)
    test_accs    = torch.zeros(niter, dtype=torch.float64)

    for it in trange(niter, desc="Training neural network..."):
        Ws[:, :, it] = W

        # forward (train)
        pre_tr  = W[:, :d] @ Xtr_t.T                     # (m, ntr)
        act_tr  = torch.clamp(pre_tr, min=0.0)
        out_tr  = (W[:, -1].unsqueeze(1) * act_tr).sum(dim=0) / m
        perf_tr = Ytr_t * out_tr

        train_losses[it] = F.softplus(-perf_tr).mean()
        margins[it]      = perf_tr.min()
        train_accs[it]   = (perf_tr > 0).to(torch.float64).mean()

        # --------- correct gradient (matches Julia): gradR = -Y * σ(-perf) / n ----------
        gradR   = -(Ytr_t * torch.sigmoid(-perf_tr)) / ntr          # (ntr,)
        gate    = (act_tr > 0).to(torch.float64)                    # (m,ntr)
        left    = W[:, -1].unsqueeze(1) * gate                      # (m,ntr)
        right   = Xtr_t * gradR.unsqueeze(1)                        # (ntr,d) elementwise
        grad_w1 = left @ right                                      # (m,d)
        grad_w2 = act_tr @ gradR                                    # (m,)
        grad    = torch.cat([grad_w1, grad_w2.unsqueeze(1)], dim=1) # (m,d+1)

        # L2 regularization on all of W (same as Julia)
        grad = grad + weight_decay * W

        # stats
        betas[it] = W.pow(2).sum() / m

        # forward (test)
        pre_te  = W[:, :d] @ Xte_t.T
        act_te  = torch.clamp(pre_te, min=0.0)
        out_te  = (W[:, -1].unsqueeze(1) * act_te).sum(dim=0) / m
        perf_te = Yte_t * out_te
        test_losses[it] = F.softplus(-perf_te).mean()
        test_accs[it]   = (perf_te > 0).to(torch.float64).mean()

        # step (Julia schedule: stepsize / sqrt(iter+1) with 1-based iter -> it+2)
        W = W - (stepsize / math.sqrt(it + 2)) * grad

    return (
        Ws.numpy(),
        train_losses.numpy(),
        margins.numpy(),
        betas.numpy(),
        train_accs.numpy(),
        test_losses.numpy(),
        test_accs.numpy(),
    )

def cluster_center(p, k):
    p = p.astype(np.int64)
    p1 = (p - 1) % k + 1
    p2 = (p - 1) // k + 1
    Δ = 1.0 / (3 * k - 1)
    x1 = Δ * (1 + 3 * (p1 - 1)) - 0.5
    x2 = Δ * (1 + 3 * (p2 - 1)) - 0.5
    return x1, x2

def illustration(k, n, m, stepsize, niter, nframes, resolution):
    # data
    Δ = 1.0 / (3 * k - 1)
    A = np.ones(k * k)
    A[np.random.permutation(k * k)[: (k * k) // 2]] = -1
    P = np.random.randint(1, k * k + 1, size=n)
    T = 2 * np.pi * np.random.rand(n)
    R = Δ * np.random.rand(n)
    x1c, x2c = cluster_center(P, k)
    Xfull = np.column_stack([np.ones(n), x1c + R * np.cos(T), x2c + R * np.sin(T)])  # (n,3)
    Yfull = A[P - 1]

    idx = np.random.permutation(n)
    ntr = int(np.floor(0.4 * n))
    tr_idx, te_idx = idx[:ntr], idx[ntr:]
    Xtr, Ytr = Xfull[tr_idx, :], Yfull[tr_idx]
    Xte, Yte = Xfull[te_idx, :], Yfull[te_idx]

    X1 = Xfull[Yfull ==  1.0, :]
    X2 = Xfull[Yfull == -1.0, :]

    Ws, train_losses, margins, betas, train_accs, test_losses, test_accs = twonet(
        Xtr, Ytr, m, stepsize, niter, weight_decay=1e-3, Xte=Xte, Yte=Yte
    )

    a = (niter - 1) / (nframes - 1) ** 4 if nframes > 1 else 0.0
    ts = np.unique((np.floor(a * (np.arange(nframes) ** 4)).astype(int) + 1))
    ts = ts[(ts >= 1) & (ts <= niter)]
    Ws = Ws[:, :, ts - 1]
    Tframes = Ws.shape[-1]

    # parameter-space projection
    Wproj = Ws[:, :3, :] * np.abs(Ws[:, 3:4, :])
    WN    = np.sqrt((Wproj ** 2).sum(axis=1, keepdims=True))
    Wdir  = Wproj / (WN + 1e-12)
    Wlog  = np.tanh(0.5 * WN) * Wdir

    # activation-space
    def activation_log(Ws_snapshots, X):
        m_, d1, T_ = Ws_snapshots.shape
        out = np.zeros((m_, 3, T_), dtype=np.float64)
        nX = X.shape[0]
        for t in range(T_):
            Wt = Ws_snapshots[:, :3, t]
            pre = Wt @ X.T
            act = np.maximum(pre, 0.0)
            M = (act @ X) / nX
            r = np.sqrt((M ** 2).sum(axis=1, keepdims=True))
            dir_ = M / (r + 1e-8)
            out[:, :, t] = np.tanh(5.0 * r) * dir_
        return out

    Wlog_act_tr = activation_log(Ws, Xtr)
    Wlog_act_te = activation_log(Ws, Xte)

    # plots
    for kf in trange(Tframes, desc="Plotting images..."):
        fig = plt.figure(figsize=(18, 8))
        gs = fig.add_gridspec(2, 3, height_ratios=[1.2, 0.9], wspace=0.35, hspace=0.35)
        indt = np.arange(0, kf + 1) if kf < 10 else np.arange(kf - 10, kf + 1)
        half = m // 2

        axP = fig.add_subplot(gs[0, 0], projection="3d")
        for i in range(Wlog.shape[0]):
            axP.plot(Wlog[i, 1, indt], Wlog[i, 2, indt], Wlog[i, 0, indt], color="k", linewidth=0.2)
        axP.plot(Wlog[:half, 1, kf], Wlog[:half, 2, kf], Wlog[:half, 0, kf], "o", color="C3", markersize=2)
        axP.plot(Wlog[half:, 1, kf], Wlog[half:, 2, kf], Wlog[half:, 0, kf], "o", color="C0", markersize=2)
        axP.set_xlim(-1,1); axP.set_ylim(-1,1); axP.set_zlim(-1,1)
        axP.set_title("Neuron trajectories (parameter)")

        axT = fig.add_subplot(gs[0, 1], projection="3d")
        for i in range(Wlog_act_tr.shape[0]):
            axT.plot(Wlog_act_tr[i, 1, indt], Wlog_act_tr[i, 2, indt], Wlog_act_tr[i, 0, indt], color="k", linewidth=0.2)
        axT.plot(Wlog_act_tr[:half, 1, kf], Wlog_act_tr[:half, 2, kf], Wlog_act_tr[:half, 0, kf], "o", color="C3", markersize=2)
        axT.plot(Wlog_act_tr[half:, 1, kf], Wlog_act_tr[half:, 2, kf], Wlog_act_tr[half:, 0, kf], "o", color="C0", markersize=2)
        axT.set_xlim(-1,1); axT.set_ylim(-1,1); axT.set_zlim(-1,1)
        axT.set_title("Neuron activation trajectories (train)")

        axE = fig.add_subplot(gs[0, 2], projection="3d")
        for i in range(Wlog_act_te.shape[0]):
            axE.plot(Wlog_act_te[i, 1, indt], Wlog_act_te[i, 2, indt], Wlog_act_te[i, 0, indt], color="k", linewidth=0.2)
        axE.plot(Wlog_act_te[:half, 1, kf], Wlog_act_te[:half, 2, kf], Wlog_act_te[:half, 0, kf], "o", color="C3", markersize=2)
        axE.plot(Wlog_act_te[half:, 1, kf], Wlog_act_te[half:, 2, kf], Wlog_act_te[half:, 0, kf], "o", color="C0", markersize=2)
        axE.set_xlim(-1,1); axE.set_ylim(-1,1); axE.set_zlim(-1,1)
        axE.set_title("Neuron activation trajectories (test)")

        axB = fig.add_subplot(gs[1, 0])
        xs = np.arange(-0.8, 0.8 + 1e-12, resolution)
        tab = np.zeros((len(xs), len(xs)))
        Wsnap = Ws[:, :, kf]
        W_in = Wsnap[:, :3]
        w_out = Wsnap[:, 3]
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
        axB.plot(X1[:, 1], X1[:, 2], "+k"); axB.plot(X2[:, 1], X2[:, 2], "_k")
        axB.set_aspect("equal"); axB.axis("off")
        axB.set_title("Decision boundary")

        axL = fig.add_subplot(gs[1, 1])
        axL.plot(np.arange(1, len(train_losses) + 1), train_losses, label="Train")
        axL.plot(np.arange(1, len(test_losses) + 1), test_losses, linestyle="--", label="Test")
        axL.axvline((ts[kf]), color="k", linestyle=":", linewidth=1)
        axL.set_xlabel("Iteration"); axL.set_ylabel("Cross Entropy Loss")
        axL.set_title("Loss (train vs test)")
        axL.grid(True); axL.legend(loc="upper right", frameon=False)

        axA = fig.add_subplot(gs[1, 2])
        axA.plot(np.arange(1, len(train_accs) + 1), train_accs, label="Train")
        axA.plot(np.arange(1, len(test_accs) + 1), test_accs, linestyle="--", label="Test")
        axA.axvline((ts[kf]), color="k", linestyle=":", linewidth=1)
        axA.set_xlabel("Iteration"); axA.set_ylabel("Accuracy"); axA.set_ylim(0, 1.0)
        axA.set_title("Accuracy (train vs test)")
        axA.grid(True); axA.legend(loc="upper right", frameon=False)

        plt.savefig(f"dynamics_wd_{kf+1}.png", bbox_inches="tight", dpi=300)
        plt.close(fig)

if __name__ == "__main__":
    k, n, m = 3, 600, 200
    stepsize, niter = 0.4, 50000
    nframes, resolution = 80, 0.02
    illustration(k, n, m, stepsize, niter, nframes, resolution)
