##############################
# Modular Addition Grokking
# (plots kept identical)
##############################

using LinearAlgebra, Random
using PyPlot, ProgressMeter
using3D()

const SEED = 0
const RNG  = MersenneTwister(SEED)

# ===============================
# TRAINING: 2-layer ReLU, multi-class softmax CE
# - Hidden weights U (m × d)
# - Class heads V (N × m)
# For plotting, we expose a single class head (vis_class) in Ws[:, end, :]
# so the original trajectory + boundary code works unchanged.
# ===============================

"""
Gradient descent for a 2-layer ReLU network trained on modular addition (multi-class) with softmax CE.
Architecture for plotting is kept effectively the same:
  - Hidden weights U (m × d) evolve exactly like before.
  - Output uses class-specific heads V (N × m) during training.
  - For plotting, we project a single chosen class head (vis_class) into the last column so the
    decision boundary & neuron trajectory visuals remain identical.

INPUT:
  Xtr         - (n_tr × d) with bias in col 1, where d=3 (bias, a, b)
  ytr         - Vector{Int} of class indices in 1..N (true sums modulo N, +1 indexing)
  m           - number of hidden neurons
  stepsize    - base step size
  niter       - iterations
KWARGS:
  N               - modulus (number of classes)
  weight_decay    - L2 decay
  Xte, yte        - optional test set
  vis_class       - which residue class to visualize for 2D boundary (0-based)

OUTPUT (same names/shapes as your original):
  Ws, train_losses, margins, betas, train_accs, test_losses, test_accs
"""
function twonet_modadd(Xtr, ytr, m, stepsize, niter; N, weight_decay=1e-4,
                       Xte=nothing, yte=nothing, vis_class=0)
    (ntr, d) = size(Xtr)  # d includes bias; keep d==3
    if Xte === nothing || yte === nothing
        Xte, yte = Xtr, ytr
    end
    nte = size(Xte, 1)

    # Hidden weights (like your W[:,1:d] before)
    U = randn(m, d)
    U[:, 1:d] .= U[:, 1:d] ./ sqrt.(sum(U[:, 1:d].^2, dims=2) .+ 1e-12)

    # Class heads (N × m)
    V = randn(N, m) / sqrt(m)

    # For plotting, we expose Ws[:,:,iter] as [U  v_vis] where v_vis is the chosen head.
    vis_idx = mod(vis_class, N) + 1

    Ws           = zeros(m, d + 1, niter)
    train_losses = zeros(niter)
    test_losses  = zeros(niter)
    margins      = zeros(niter)  # average (true logit - best other)
    betas        = zeros(niter)
    train_accs   = zeros(niter)
    test_accs    = zeros(niter)

    @showprogress 1 "Training (modular addition)..." for iter = 1:niter
        # ----- store for plotting -----
        Ws[:, 1:d, iter] = U
        Ws[:, end, iter] = @view(V[vis_idx, :])[:]  # chosen class head as the plotting column

        # ----- forward (train) -----
        Htr = max.(U * Xtr', 0.0)             # (m × ntr)
        logits = (V * Htr) ./ m               # (N × ntr)

        # softmax CE
        logits_shift = logits .- maximum(logits, dims=1)
        exps = exp.(logits_shift)
        Z = sum(exps, dims=1)
        probs = exps ./ Z

        # loss
        loss = 0.0
        @inbounds for i in 1:ntr
            loss -= log(probs[ytr[i], i] + 1e-12)
        end
        train_losses[iter] = loss / ntr

        # accuracy
        pred_tr = vec(map(argmax, eachcol(probs)))
        train_accs[iter] = mean(pred_tr .== ytr)

        # multiclass margin (avg)
        margin_sum = 0.0
        @inbounds for i in 1:ntr
            yi = ytr[i]
            true_logit = logits[yi, i]
            # max over others
            max_other = -Inf
            @inbounds for c in 1:N
                c == yi && continue
                v = logits[c, i]
                if v > max_other
                    max_other = v
                end
            end
            margin_sum += (true_logit - max_other)
        end
        margins[iter] = margin_sum / ntr

        # ----- backward -----
        # dL/dlogits = probs - onehot
        G = copy(probs)
        @inbounds for i in 1:ntr
            G[ytr[i], i] -= 1.0
        end
        G ./= ntr                           # (N × ntr)

        # grads for V and U
        gradV = (G * Htr') ./ m             # (N × m)
        dH = V' * G                          # (m × ntr)
        dH .= dH .* (Htr .> 0.0)            # ReLU backprop
        gradU = dH * Xtr                    # (m × d)

        # L2
        gradU .+= weight_decay .* U
        gradV .+= weight_decay .* V

        # track squared norm (both U and V)
        betas[iter] = (sum(U.^2) + sum(V.^2)) / m

        # ----- forward (test) -----
        Hte = max.(U * Xte', 0.0)
        logits_te = (V * Hte) ./ m
        logits_te_shift = logits_te .- maximum(logits_te, dims=1)
        exps_te = exp.(logits_te_shift)
        Zte = sum(exps_te, dims=1)
        probs_te = exps_te ./ Zte

        # test loss/acc
        loss_te = 0.0
        @inbounds for i in 1:nte
            loss_te -= log(probs_te[yte[i], i] + 1e-12)
        end
        test_losses[iter] = loss_te / nte
        pred_te = vec(map(argmax, eachcol(probs_te)))
        test_accs[iter] = mean(pred_te .== yte)

        # ----- step -----
        η = stepsize / sqrt(iter + 1)
        U .-= η .* gradU
        V .-= η .* gradV
    end

    return Ws, train_losses, margins, betas, train_accs, test_losses, test_accs
end

# ===============================
# DATA: modular addition in 2D (bias + two coords), so boundary plot works
# ===============================

"Build the full modular addition table, encode to 2D + bias so plots stay the same."
function make_modadd_dataset(N)
    # all pairs (a,b)
    A = repeat(collect(0:N-1), inner=N)
    B = repeat(collect(0:N-1), outer=N)
    # normalize to roughly [-0.5, 0.5] like a unit box for plotting
    x1 = A ./ (N - 1) .- 0.5
    x2 = B ./(N - 1) .- 0.5
    X = hcat(ones(length(x1)), x1, x2)  # (n × 3), bias first

    y = mod.(A .+ B, N) .+ 1            # classes in 1..N (Julia 1-based)
    return X, y
end

# ===============================
# PLOTTING DRIVER (identical layout to your original)
# ===============================

"""
Visualizes:
  (1) Neuron trajectories in parameter space (unchanged)
  (2) Decision boundary in input space (for one selected residue class vs rest)
  (3) Train vs Test loss
  (4) Train vs Test accuracy

We construct a held-out test set by random split of a sampled subset of the full table.
"""
function illustration_modadd(N, n, m, stepsize, niter, nframes, resolution;
                             weight_decay=1e-3, vis_class=0, split=0.4)
    # --- data ---
    # build full table, then sample without replacement
    Xfull_all, yfull_all = make_modadd_dataset(N)
    nmax  = size(Xfull_all, 1)               # = N^2
    n_eff = min(n, nmax)                     # cap to available rows
    if n_eff < n
        @warn "Requested n=$n exceeds table size N^2=$nmax; using n_eff=$n_eff"
    end

    idx_all = randperm(RNG, nmax)
    idx = idx_all[1:n_eff]
    Xfull = Xfull_all[idx, :]
    yfull = yfull_all[idx]

    # --- split (train/test) using the randomized order already in Xfull/yfull
    ntr = max(1, Int(floor(split * n_eff)))
    Xtr, ytr = Xfull[1:ntr, :],            yfull[1:ntr]
    Xte, yte = Xfull[ntr+1:end, :],        yfull[ntr+1:end]

    # For boundary scatter: positives = chosen residue, negatives = rest
    vis_idx = mod(vis_class, N) + 1
    yvis_full = (yfull .== vis_idx)
    X1 = Xfull[yvis_full, :]
    X2 = Xfull[.!yvis_full, :]

    # --- training ---
    Ws, train_losses, margins, betas, train_accs, test_losses, test_accs =
        twonet_modadd(Xtr, ytr, m, stepsize, niter;
                      N=N, weight_decay=weight_decay, Xte=Xte, yte=yte, vis_class=vis_class)

    # time subsampling as before
    a  = (niter - 1) / (nframes - 1)^4
    ts = unique(Int.(floor.(a .* (0:nframes-1).^4)) .+ 1)
    ts = ts[ts .>= 1 .&& ts .<= niter]
    Ws = Ws[:, :, ts]

    # projection of hidden neurons (unchanged)
    Wproj = Ws[:, 1:end-1, :] .* abs.(Ws[:, end:end, :])
    WN    = sqrt.(sum(Wproj.^2, dims=2))
    Wdir  = Wproj ./ (WN .+ 1e-12)
    Wlog  = tanh.(0.5 .* WN) .* Wdir

    # --- plotting loop (identical structure & styles) ---
    @showprogress 1 "Plotting images..." for kf = 1:length(ts)
        ioff()
        fig = figure(figsize = [14, 8])

        gs = fig.add_gridspec(2, 2, height_ratios=[1.2, 0.9], wspace=0.3, hspace=0.35)

        # (1) neuron dynamics (top-left)
        ax1 = fig.add_subplot(gs[1, 1], projection="3d")
        indt = kf < 11 ? (1:kf) : ((kf-10):kf)
        for i = 1:size(Wlog, 1)
            plot3D(Wlog[i,2,indt], Wlog[i,3,indt], Wlog[i,1,indt], color="k", linewidth=0.2)
        end
        plot3D(Wlog[1:div(size(Wlog,1),2),2,kf],Wlog[1:div(size(Wlog,1),2),3,kf],Wlog[1:div(size(Wlog,1),2),1,kf],"o",color="C3",markersize=1)
        plot3D(Wlog[div(size(Wlog,1),2)+1:end,2,kf],Wlog[div(size(Wlog,1),2)+1:end,3,kf],Wlog[div(size(Wlog,1),2)+1:end,1,kf],"o",color="C0",markersize=1)
        ax1.set_xlim3d(-1,1); ax1.set_ylim3d(-1,1); ax1.set_zlim3d(-1,1)
        ax1.set_title("Neuron trajectories")

        # (2) decision boundary (top-right), f uses the visualized class head
        ax2 = fig.add_subplot(gs[1, 2])
        f(x1,x2,kk) = (1/size(Ws,1)) * sum(Ws[:,end,kk] .* max.(Ws[:,1:3,kk]*[1;x1;x2],0.0))
        xs = -0.8:resolution:0.8
        tab = [f(xs[i],xs[j],kf) for i=1:length(xs), j=1:length(xs)]
        pcolormesh(xs', xs, tanh.(tab'), cmap="coolwarm", shading="gouraud",
                   vmin=-1.0, vmax=1.0, edgecolor="face")
        contour(xs', xs, tanh.(tab'), levels=0, colors="k", linewidths=2)
        plot(X1[:,2],X1[:,3],"+k"); plot(X2[:,2],X2[:,3],"_k")
        axis("equal"); axis("off")
        ax2.set_title("Decision boundary (class $(vis_class))")

        # (3) train vs test loss (bottom-left)
        ax3 = fig.add_subplot(gs[2, 1])
        ax3.plot(1:length(train_losses), train_losses, "C2", linewidth=1.5, label="Train")
        ax3.plot(1:length(test_losses),  test_losses,  "C4", linewidth=1.2, linestyle="--", label="Test")
        ax3.axvline(ts[kf], color="k", linestyle=":", linewidth=1)
        ax3.set_xlabel("Iteration")
        ax3.set_ylabel("Cross Entropy Loss")
        ax3.set_title("Loss (train vs test)")
        ax3.grid(true)
        ax3.legend(loc="upper right", frameon=false)

        # (4) train vs test accuracy (bottom-right)
        ax4 = fig.add_subplot(gs[2, 2])
        ax4.plot(1:length(train_accs), train_accs, "C1", linewidth=1.5, label="Train")
        ax4.plot(1:length(test_accs),  test_accs,  "C0", linewidth=1.2, linestyle="--", label="Test")
        ax4.axvline(ts[kf], color="k", linestyle=":", linewidth=1)
        ax4.set_xlabel("Iteration")
        ax4.set_ylabel("Accuracy")
        ax4.set_ylim(0, 1.0)
        ax4.set_title("Accuracy (train vs test)")
        ax4.grid(true)
        ax4.legend(loc="upper right", frameon=false)

        savefig("dynamics_modadd_$(kf).png", bbox_inches="tight", dpi=300)
        close(fig)
    end
end

# Convenience wrapper so you can call with keywords (what you tried earlier)
function illustration_modadd(; N, n, m, stepsize, niter, nframes, resolution,
                              weight_decay=1e-3, vis_class=0, split=0.4)
    illustration_modadd(N, n, m, stepsize, niter, nframes, resolution;
                        weight_decay=weight_decay, vis_class=vis_class, split=split)
end

# ===============================
# Example usage (uncomment to run)
# ===============================
# illustration_modadd(N=97, n=9409, m=512, stepsize=0.1,
#                     niter=20_000, nframes=80, resolution=0.01,
#                     weight_decay=1e-3, vis_class=0, split=0.4)
