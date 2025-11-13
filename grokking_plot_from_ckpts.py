import os
import math
import numpy as np
import torch
from tqdm import trange
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import os
import re
import torch

CKPT_PATTERN = re.compile(r"^ckpt_step(\d+)\.pt$")

def find_checkpoints(ckpt_dir):
    """
    Return sorted list of (step, filename) checkpoint files.
    """
    checkpoints = []
    for fname in os.listdir(ckpt_dir):
        m = CKPT_PATTERN.match(fname)
        if m:
            step = int(m.group(1))
            checkpoints.append((step, fname))

    if not checkpoints:
        raise FileNotFoundError(f"No ckpt_step*.pt files found in {ckpt_dir}")

    checkpoints.sort(key=lambda x: x[0])
    return checkpoints


def verify_consecutive_steps(checkpoints):
    """
    Ensure checkpoints have strictly consecutive steps (constant step size, no gaps).
    """
    steps = [s for s, _ in checkpoints]

    if len(steps) == 1:
        return 0  # trivial

    diffs = [b - a for a, b in zip(steps[:-1], steps[1:])]
    expected_diff = diffs[0]

    for i, d in enumerate(diffs):
        if d != expected_diff:
            raise ValueError(
                f"Non-consecutive steps between {steps[i]} and {steps[i+1]} "
                f"(diff={d}, expected={expected_diff})."
            )

    if expected_diff <= 0:
        raise ValueError(f"Invalid step size {expected_diff}")

    return expected_diff


def load_checkpoints(ckpt_dir):
    """
    Finds, checks, and loads all checkpoints in a directory.
    Returns list of (step, checkpoint_data).
    """
    checkpoints = find_checkpoints(ckpt_dir)
    step_size = verify_consecutive_steps(checkpoints)

    print(f"Found {len(checkpoints)} checkpoints.")
    print("Steps:", [s for s, _ in checkpoints])
    if step_size:
        print(f"Inferred step size: {step_size}")
    else:
        print("Only one checkpoint — no step size to infer.")

    loaded = []
    for step, fname in checkpoints:
        path = os.path.join(ckpt_dir, fname)
        print(f"Loading {path} ...")
        ckpt = torch.load(path, map_location="cpu")
        loaded.append((step, ckpt))

    print("Done!")
    return loaded


# -----------------------
# Precision / RNG
# -----------------------
SEED = 0
torch.set_default_dtype(torch.float64)
torch.manual_seed(SEED)
np.random.seed(SEED)


# -----------------------
# Build Ws from checkpoints
# -----------------------
def build_Ws_from_ckpts(ckpts, n_components=3, out_index=0):
    """
    Build Ws (m, 3+1, T) from a list of checkpoints.

    Checkpoint structure is assumed to be:
        ckpts[t][1]["model_state_dict"]["fc1.weight"]  -> (m, 6)
        ckpts[t][1]["model_state_dict"]["fc_out.weight"] -> (2, m)

    We do:
      - fc1.weight: (m, 6) -> project to (m, 3) in a global basis (SVD over all ckpts)
      - fc_out.weight: (2, m) -> take one row (out_index) as w_out: (m,)

    Returns:
        Ws:    (m, 4, T) with cols [z1, z2, z3, w_out]
        steps: (T,) array of training steps (from ckpts[i][0])
    """
    T = len(ckpts)
    if T == 0:
        raise ValueError("ckpts is empty; did you load any checkpoints?")

    # Grab first checkpoint to infer shapes
    first_state_dict = ckpts[0][1]["model_state_dict"]
    W1_0 = first_state_dict["fc1.weight"].detach().cpu().numpy()  # (m, 6)
    m, d_in = W1_0.shape

    if d_in != 6:
        raise ValueError(f"Expected fc1.weight to have shape (m, 6), got (m, {d_in})")

    # Stack all fc1 weights over time: (T, m, 6)
    W1_all = np.zeros((T, m, d_in), dtype=np.float64)
    steps = np.zeros(T, dtype=np.int64)

    for t, (step, state) in enumerate(ckpts):
        steps[t] = int(step)
        W1_t = state["model_state_dict"]["fc1.weight"].detach().cpu().numpy()  # (m, 6)
        if W1_t.shape != (m, d_in):
            raise ValueError(
                f"fc1.weight shape mismatch at checkpoint {t}: "
                f"expected {(m, d_in)}, got {W1_t.shape}"
            )
        W1_all[t] = W1_t

    # Flatten for SVD: each neuron's 6-dim weight vector at each time is a sample
    X = W1_all.reshape(T * m, d_in)  # (T*m, 6)

    # Center
    X_mean = X.mean(axis=0, keepdims=True)   # (1, 6)
    Xc = X - X_mean

    # SVD to get a global 6 -> 3 projection basis
    # Xc ≈ U S V^T, with V: (6, 6)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    V = Vt.T                      # (6, 6)
    P = V[:, :n_components]       # (6, 3) projection matrix

    # Allocate Ws: (m, 3 coords + 1 output weight, T)
    Ws = np.zeros((m, n_components + 1, T), dtype=np.float64)

    for t, (_, state) in enumerate(ckpts):
        # ----- fc1.weight: project 6D -> 3D -----
        W1 = state["model_state_dict"]["fc1.weight"].detach().cpu().numpy()  # (m, 6)
        W1c = W1 - X_mean               # center with global mean (broadcasts (1, 6) to (m, 6))
        Z = W1c @ P                     # (m, 3)

        # ----- fc_out.weight: (2, m) -> choose one dim -----
        Wout_full = state["model_state_dict"]["fc_out.weight"].detach().cpu().numpy()  # (2, m)
        if Wout_full.shape[0] <= out_index:
            raise ValueError(
                f"Requested out_index={out_index}, but fc_out.weight has shape {Wout_full.shape}"
            )
        w_out = Wout_full[out_index, :]  # (m,)

        Ws[:, :n_components, t] = Z
        Ws[:, n_components, t] = w_out

    return Ws, steps


# -----------------------
# Visualization from Ws
# -----------------------
def illustration_ckpts(ckpts, nframes=200, out_index=0, save_dir="."):
    """
    Visualize neuron trajectories using pre-trained checkpoints.

    Uses:
      - Ws[:, :3, t] as 3D coords (from reduced fc1 weights)
      - Ws[:, 3, t]  as w_out (truncated fc_out row)

    Plots 3D parameter-space trajectories with scaling by |w_out|.
    Saves PNGs to save_dir/dynamics_ckpt_XXXX.png.
    """
    os.makedirs(save_dir, exist_ok=True)

    # Build Ws from checkpoints
    Ws, steps = build_Ws_from_ckpts(ckpts, n_components=3, out_index=out_index)  # (m, 4, T)
    m, _, T = Ws.shape

    # Time subsampling (quartic spacing) over available checkpoints
    if nframes > T:
        nframes = T
    if nframes < 1:
        raise ValueError("nframes must be >= 1")

    a = (T - 1) / (nframes - 1) ** 4 if nframes > 1 else 0.0
    ts_idx = np.unique(
        np.floor(a * (np.arange(nframes) ** 4)).astype(int)
    )
    ts_idx = ts_idx[(ts_idx >= 0) & (ts_idx < T)]
    Ws = Ws[:, :, ts_idx]        # (m, 4, Tframes)
    steps = steps[ts_idx]
    Tframes = Ws.shape[-1]

    # Parameter-space projection (data-agnostic),
    # mirroring the original code: scale by |w_out|, then squash with tanh
    Wproj = Ws[:, :3, :] * np.abs(Ws[:, 3:4, :])               # (m, 3, T)
    WN = np.sqrt((Wproj ** 2).sum(axis=1, keepdims=True))      # (m, 1, T)
    Wdir = Wproj / (WN + 1e-12)                                # (m, 3, T)
    #Wlog = np.tanh(0.5 * WN) * Wdir                            # (m, 3, T)
    Wlog = Wdir

    for kf in trange(Tframes, desc="Plotting ckpt images..."):
        fig = plt.figure(figsize=(6, 6))
        axP = fig.add_subplot(111, projection="3d")

        # show last few steps as trajectories
        if kf < 10:
            indt = np.arange(0, kf + 1)
        else:
            indt = np.arange(kf - 10, kf + 1)

        # trajectory lines
        for i in range(Wlog.shape[0]):
            axP.plot(
                Wlog[i, 1, indt],  # x ~ w1
                Wlog[i, 2, indt],  # y ~ w2
                Wlog[i, 0, indt],  # z ~ "bias-like"
                linewidth=0.2
            )

        # final positions at this frame, split by sign of w_out
        w_out_now = Ws[:, 3, kf]
        pos_idx = w_out_now >= 0
        neg_idx = ~pos_idx

        axP.plot(
            Wlog[pos_idx, 1, kf],
            Wlog[pos_idx, 2, kf],
            Wlog[pos_idx, 0, kf],
            "o",
            markersize=2,
            label="w_out >= 0" if kf == 0 else None,
        )
        axP.plot(
            Wlog[neg_idx, 1, kf],
            Wlog[neg_idx, 2, kf],
            Wlog[neg_idx, 0, kf],
            "^",
            markersize=2,
            label="w_out < 0" if kf == 0 else None,
        )

        axP.set_xlim(-1, 1)
        axP.set_ylim(-1, 1)
        axP.set_zlim(-1, 1)
        axP.set_xlabel("dim 1 (proj)")
        axP.set_ylabel("dim 2 (proj)")
        axP.set_zlabel("dim 3 (proj)")
        axP.set_title(f"Neuron trajectories\n(ckpt step = {steps[kf]})")

        if kf == 0:
            axP.legend(loc="upper right", frameon=False)

        fname = os.path.join(save_dir, f"dynamics_ckpt_{kf+1:04d}.png")
        plt.savefig(fname, bbox_inches="tight", dpi=300)
        plt.close(fig)


# -----------------------
# Main
# -----------------------
if __name__ == "__main__":
    # Path where your checkpoints live
    ckpt_dir = "/n/netscratch/dam_lab/Lab/iglee/grokking_exp_1112_small_parity/sparse_parity/seed_826/lr_0.01_wd_0.01/checkpoints/"

    # This should be your existing helper that returns a list-like structure where
    #   ckpts[i][1]["model_state_dict"]["fc1.weight"]  -> (200, 6)
    #   ckpts[i][1]["model_state_dict"]["fc_out.weight"] -> (2, 200)
    ckpts = load_checkpoints(ckpt_dir)

    # Number of frames to render (will be capped at number of checkpoints)
    nframes = 500

    # Which output dimension from fc_out.weight to use (0 or 1)
    out_index = 0

    # Directory to save PNGs into
    save_dir = "ckpt_dynamics"

    illustration_ckpts(ckpts, nframes=nframes, out_index=out_index, save_dir=save_dir)
