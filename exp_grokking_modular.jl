using LinearAlgebra, Random
using PyPlot, ProgressMeter
using3D()

const SEED = 0
const RNG  = MersenneTwister(SEED)

"""
Gradient descent to train a 2-layer ReLU neural network for the exponential/logistic-type loss.

We minimize  L = (1/n) * sum_i log(1 + exp(-y_i f(x_i)))  (cross-entropy style)

INPUT:
  Xtr, Ytr  - training data (n_tr × d+1) with bias in col 1, labels in {-1,+1}
  m         - number of hidden neurons
  stepsize  - base step size
  niter     - number of iterations

KWARGS:
  weight_decay - L2 weight decay (default 1e-4)
  Xte, Yte     - (optional) test set; if not provided, uses training set

OUTPUT:
  Ws             - trajectory of weights
  train_losses   - training loss values
  margins        - min margin over training samples
  betas          - average squared weight norm
  train_accs     - training accuracy per iter
  test_losses    - test loss values
  test_accs      - test accuracy per iter
"""
function twonet(Xtr, Ytr, m, stepsize, niter; weight_decay=1e-4, Xte=nothing, Yte=nothing)
    (ntr, d) = size(Xtr)              # d INCLUDES the bias column
    if Xte === nothing || Yte === nothing
        Xte, Yte = Xtr, Ytr
    end
    nte = size(Xte, 1)

    # initialize weights: rows = neurons, cols = (input block of size d) + (output weight)
    W = randn(m, d + 1)
    W[:, 1:d] = W[:, 1:d] ./ sqrt.(sum(W[:, 1:d].^2, dims=2))
    W[1:div(m,2), end]       .=  1.0
    W[div(m,2)+1:end, end]   .= -1.0

    Ws           = zeros(m, d + 1, niter)
    train_losses = zeros(niter)
    test_losses  = zeros(niter)
    margins      = zeros(niter)
    betas        = zeros(niter)
    train_accs   = zeros(niter)
    test_accs    = zeros(niter)

    @showprogress 1 "Training neural network..." for iter = 1:niter
        Ws[:, :, iter] = W

        # ---------- forward (train) ----------
        act_tr  = max.(W[:, 1:end-1] * Xtr', 0.0)                   # (m × ntr), OK: (m×d)*(d×ntr)
        out_tr  = (1/m) * sum(W[:, end] .* act_tr, dims=1)[:]       # (ntr)
        perf_tr = Ytr .* out_tr

        train_losses[iter] = sum(log.(1 .+ exp.(-perf_tr))) / ntr
        margins[iter]      = minimum(perf_tr)
        train_accs[iter]   = sum(perf_tr .> 0) / ntr

        # grad (train)
        gradR   = -Ytr .* (1 .- 1 ./ (1 .+ exp.(-perf_tr))) / ntr
        grad_w1 = (W[:, end] .* float.(act_tr .> 0)) * (Xtr .* gradR)  # (m×ntr)*(ntr×d) -> (m×d)
        grad_w2 = act_tr * gradR                                       # (m×ntr)*(ntr)   -> (m)
        grad    = cat(grad_w1, grad_w2, dims=2)

        # L2 regularization
        grad += weight_decay * W

        # track squared norm
        betas[iter] = sum(W.^2) / m

        # ---------- forward (test) ----------
        act_te  = max.(W[:, 1:end-1] * Xte', 0.0)
        out_te  = (1/m) * sum(W[:, end] .* act_te, dims=1)[:]
        perf_te = Yte .* out_te
        test_losses[iter] = sum(log.(1 .+ exp.(-perf_te))) / nte
        test_accs[iter]   = sum(perf_te .> 0) / nte

        # step
        W = W - stepsize * grad / sqrt(iter + 1)
    end

    return Ws, train_losses, margins, betas, train_accs, test_losses, test_accs
end


"Coordinates of the 2D cluster centers"
function cluster_center(p, k)
    p1 = mod.(p .- 1, k) .+ 1
    p2 = div.(p .- 1, k) .+ 1
    Δ = 1 / (3k - 1)
    x1 = Δ * (1 .+ 3 .* (p1 .- 1)) .- 1/2
    x2 = Δ * (1 .+ 3 .* (p2 .- 1)) .- 1/2
    return x1, x2
end


"""
Visualizes:
  (1) Neuron trajectories in parameter space
  (2) Decision boundary in input space
  (3) Train vs Test loss (bottom-left)
  (4) Train vs Test accuracy (bottom-right)

NOTE: We generate `n` total samples and do an 80/20 train/test split.
"""
function illustration(k, n, m, stepsize, niter, nframes, resolution)
    # --- data generation ---
    Δ = 1 / (3k - 1)
    A = ones(k^2)
    A[randperm(k^2)[1:div(k^2, 2)]] .= -1

    P = rand(1:k^2, n)
    T = 2π * rand(n)
    R = Δ * rand(n)
    Xfull = cat(ones(n),
                cluster_center(P, k)[1] .+ R .* cos.(T),
                cluster_center(P, k)[2] .+ R .* sin.(T);
                dims = 2)
    Yfull = A[P]

    # 80/20 split (held-out test set) - for typical experiments (runs up to 7)
    # 20/80 split for grokking settings - for runs 7,8
    # 40/60 split for runs 9

    idx = randperm(n)
    ntr = Int(floor(0.4n))
    tr_idx, te_idx = idx[1:ntr], idx[ntr+1:end]
    Xtr, Ytr = Xfull[tr_idx, :], Yfull[tr_idx]
    Xte, Yte = Xfull[te_idx, :], Yfull[te_idx]

    # For boundary scatter (show both classes, all points)
    X1 = Xfull[Yfull .== 1, :]
    X2 = Xfull[Yfull .== -1, :]

    # --- training ---
    Ws, train_losses, margins, betas, train_accs, test_losses, test_accs =
        twonet(Xtr, Ytr, m, stepsize, niter; weight_decay=1e-3, Xte=Xte, Yte=Yte)

    # time sampling for visualization
    a  = (niter - 1) / (nframes - 1)^4
    ts = unique(Int.(floor.(a .* (0:nframes-1).^4)) .+ 1)
    ts = ts[ts .>= 1 .&& ts .<= niter]
    Ws = Ws[:, :, ts]

    # projection of hidden neurons
    Wproj = Ws[:, 1:end-1, :] .* abs.(Ws[:, end:end, :])
    WN    = sqrt.(sum(Wproj.^2, dims=2))
    Wdir  = Wproj ./ WN
    Wlog  = tanh.(0.5 .* WN) .* Wdir

    # --- plotting loop ---
    @showprogress 1 "Plotting images..." for kf = 1:length(ts)
        ioff()
        fig = figure(figsize = [14, 8])

        # 2×2 grid: top row (trajectories, boundary), bottom row (loss, accuracy)
        gs = fig.add_gridspec(2, 2, height_ratios=[1.2, 0.9], wspace=0.3, hspace=0.35)

        # (1) neuron dynamics (top-left)
        ax1 = fig.add_subplot(gs[1, 1], projection="3d")
        indt = kf < 11 ? (1:kf) : ((kf-10):kf)
        for i = 1:size(Wlog, 1)
            plot3D(Wlog[i,2,indt], Wlog[i,3,indt], Wlog[i,1,indt], color="k", linewidth=0.2)
        end
        plot3D(Wlog[1:div(m,2),2,kf],Wlog[1:div(m,2),3,kf],Wlog[1:div(m,2),1,kf],"o",color="C3",markersize=1)
        plot3D(Wlog[div(m,2)+1:end,2,kf],Wlog[div(m,2)+1:end,3,kf],Wlog[div(m,2)+1:end,1,kf],"o",color="C0",markersize=1)
        ax1.set_xlim3d(-1,1); ax1.set_ylim3d(-1,1); ax1.set_zlim3d(-1,1)
        ax1.set_title("Neuron trajectories")

        # (2) decision boundary (top-right)
        ax2 = fig.add_subplot(gs[1, 2])
        f(x1,x2,kk) = (1/m) * sum(Ws[:,end,kk] .* max.(Ws[:,1:3,kk]*[1;x1;x2],0.0))
        xs = -0.8:resolution:0.8
        tab = [f(xs[i],xs[j],kf) for i=1:length(xs), j=1:length(xs)]
        pcolormesh(xs', xs, tanh.(tab'), cmap="coolwarm", shading="gouraud",
                   vmin=-1.0, vmax=1.0, edgecolor="face")
        contour(xs', xs, tanh.(tab'), levels=0, colors="k", linewidths=2)
        plot(X1[:,2],X1[:,3],"+k"); plot(X2[:,2],X2[:,3],"_k")
        axis("equal"); axis("off")
        ax2.set_title("Decision boundary")

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

        savefig("dynamics_wd_$(kf).png", bbox_inches="tight", dpi=300)
        close(fig)
    end
end
