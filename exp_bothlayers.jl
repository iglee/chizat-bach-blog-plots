# copyright Lénaïc Chizat
# the code shows the training dynamics in parameter and predictor space for a 2-layer relu NN on a two class classification task
# we optimize the exponential loss (the behavior is sensibly the same with the logistic loss)
# we use a specific step-size schedule motivated by the theory in this ref https://arxiv.org/abs/2002.04486 (sections 4 & 5)

# HOW TO RUN THE CODE
# in a prompt first run:
# include("exp_bothlayers.jl")
# Then for fast prototyping, run for instance the following code (takes 1 min)
# illustration(4, 60, 200, 0.4, 100000, 10, 0.05);
# These parameters are explained in the header of the function "illustration"
# A good illustration is obtained as follows (takes 20 min):
# illustration(4, 60, 400, 0.4, 1000000, 400, 0.005);


using LinearAlgebra, Random
using PyPlot, ProgressMeter
using3D()

const SEED = 0
const RNG  = MersenneTwister(SEED)

"""
Gradient descent to train a 2-layer ReLU neural network for the exponential loss.

We minimize  L = (1/n) * sum_i exp(-y_i f(x_i))

INPUT:
  X        - training input, shape (n, d)
  Y        - training labels in {-1, +1}
  m        - number of hidden neurons
  stepsize - base step size
  niter    - number of iterations

OUTPUT:
  Ws       - trajectory of weights
  losses   - training loss values (decreasing)
  margins  - minimum margin over samples
  betas    - average squared weight norm
"""
function twonet(X, Y, m, stepsize, niter; weight_decay=1e-4)
    (n, d) = size(X)

    # initialize weights: rows = neurons, cols = input dims + output weight
    W = randn(m, d + 1)
    W[:, 1:d] = W[:, 1:d] ./ sqrt.(sum(W[:, 1:d].^2, dims=2)) # normalize inputs
    W[1:div(m,2), end]       .=  1.0
    W[div(m,2)+1:end, end]   .= -1.0

    Ws      = zeros(m, d + 1, niter)
    losses  = zeros(niter)
    margins = zeros(niter)
    betas   = zeros(niter)

    @showprogress 1 "Training neural network..." for iter = 1:niter
        Ws[:, :, iter] = W

        # Forward pass
        act  = max.(W[:, 1:end-1] * X', 0.0)             # (m × n)
        out  = (1/m) * sum(W[:, end] .* act, dims=1)[:]  # (n)
        perf = Y .* out                                  # margins y_i f(x_i)

        # Cross-entropy loss (mean over samples)
        losses[iter] = sum(log.(1 .+ exp.(-perf))) / n
        margins[iter] = minimum(perf)

        # Gradient of loss w.r.t. perf
        gradR = -Y .* (1 .- 1 ./ (1 .+ exp.(-perf))) / n  # = -Y * sigmoid(-perf) / n

        # Gradients
        grad_w1 = (W[:, end] .* float.(act .> 0)) * (X .* gradR)
        grad_w2 = act * gradR
        grad = cat(grad_w1, grad_w2, dims=2)

        # Add L2 regularization (weight decay)
        grad += weight_decay * W

        # Store average squared norm
        betas[iter] = sum(W.^2) / m

        # Gradient descent step
        W = W - stepsize * grad / sqrt(iter + 1)
    end

    return Ws, losses, margins, betas
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
  (3) Training loss curve
"""
function illustration(k, n, m, stepsize, niter, nframes, resolution)
    # --- data generation ---
    Δ = 1 / (3k - 1)
    A = ones(k^2)
    A[randperm(k^2)[1:div(k^2, 2)]] .= -1

    P = rand(1:k^2, n)
    T = 2π * rand(n)
    R = Δ * rand(n)
    X = cat(ones(n),
            cluster_center(P, k)[1] .+ R .* cos.(T),
            cluster_center(P, k)[2] .+ R .* sin.(T);
            dims = 2)
    Y = A[P]

    X1 = X[Y .== 1, :]
    X2 = X[Y .== -1, :]

    # --- training ---
    Ws, losses, margins, betas = twonet(X, Y, m, stepsize, niter; weight_decay=1e-4)

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
        fig = figure(figsize = [16, 4])
        # define a 1×3 grid with custom relative widths (make loss plot wider)
        gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 1.6], wspace=0.25)

        # (1) neuron dynamics
        ax1 = fig.add_subplot(gs[1], projection="3d")
        if kf < 11
            indt = 1:kf
        else
            indt = (kf-10):kf
        end
        for i = 1:size(Wlog, 1)
            plot3D(Wlog[i,2,indt], Wlog[i,3,indt], Wlog[i,1,indt], color="k", linewidth=0.2)
        end
        plot3D(Wlog[1:div(m,2),2,kf],Wlog[1:div(m,2),3,kf],Wlog[1:div(m,2),1,kf],"o",color="C3",markersize=1)
        plot3D(Wlog[div(m,2)+1:end,2,kf],Wlog[div(m,2)+1:end,3,kf],Wlog[div(m,2)+1:end,1,kf],"o",color="C0",markersize=1)
        ax1.set_xlim3d(-1,1); ax1.set_ylim3d(-1,1); ax1.set_zlim3d(-1,1)
        ax1.set_title("Neuron trajectories")

        # (2) decision boundary
        ax2 = fig.add_subplot(gs[2])
        f(x1,x2,kk) = (1/m) * sum(Ws[:,end,kk] .* max.(Ws[:,1:3,kk]*[1;x1;x2],0.0))
        xs = -0.8:resolution:0.8
        tab = [f(xs[i],xs[j],kf) for i=1:length(xs), j=1:length(xs)]
        pcolormesh(xs', xs, tanh.(tab'), cmap="coolwarm", shading="gouraud",
                   vmin=-1.0, vmax=1.0, edgecolor="face")
        contour(xs', xs, tanh.(tab'), levels=0, colors="k", linewidths=2)
        plot(X1[:,2],X1[:,3],"+k"); plot(X2[:,2],X2[:,3],"_k")
        axis("equal"); axis("off")
        ax2.set_title("Decision boundary")

        # (3) true training loss
        ax3 = fig.add_subplot(gs[3])
        ax3.plot(1:niter, losses, "C2", linewidth=1.5)
        ax3.axvline(ts[kf], color="k", linestyle="--", linewidth=1)
        ax3.set_xlabel("Iteration")
        ax3.set_ylabel("Cross Entropy Loss")
        ax3.set_title("Training loss")
        ax3.grid(true)

        savefig("dynamics_wd_$(kf).png", bbox_inches="tight", dpi=300)
        close(fig)
    end
end