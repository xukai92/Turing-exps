# Load packages
using Distributions, Turing

# Load dataset
include("timeseries.data.jl")

# Compute initial state (initial), transition matrix (T)
# and means (m) for non-Bayesian Hidden Markov Model
initial = fill(1.0 / K, K)

T = zeros(K,K); for i=1:K; T[i,:] = rand(Dirichlet(ones(K)./K)); end
T = T + K*eye(K)/K; for i=1:K; T[i,:] = T[i,:] ./ sum(T[i,:]); end

m = (collect(1.0:K)*2-K)*2

# Define a Hidden Markov Model model with parameters:
#   N   -   number of observations
#   K   -   number of states
#   y   -   observations
#   T   -   transition matrix
#   m   -   means
@model HMM(N, K, y, T, m) = begin
    s = tzeros(Int, N)
    s[1] ~ Categorical(fill(1.0 / K, K))
    for i = 2:N
        s[i] ~ Categorical(vec(T[s[i-1],:]))
        y[i] ~ Normal(m[s[i]], 1)
    end
end

# Collect 500 samples using the Particle Gibbs sampler (50 particles used)
samples = sample(HMM(N, K, y, T, m), PG(50, 500));



# Visualization
using PyPlot, PyCall
@pyimport matplotlib.animation as animation
plt["style"]["use"]("ggplot")

# Animation set-up
fig, ax = plt[:subplots]();
ax[:set_xlim](( 0, N))
ax[:set_ylim]((-20, 20))
line, = ax[:plot]([], [], lw=2)

# Initialization function: plot the background of each frame
function init()
    ax[:plot](1:N, y, "rs")
    return (line,)
end

# Animation function. This is called sequentially
function animate(i)
    n = mod(i,length(samples.value))
    line[:set_data](1:N, m[samples[:s][n+1]])
    return (line,)
end

# Call the animator. blit=True means only re-draw the parts that have changed.
anim = animation.FuncAnimation(fig, animate, init_func=init,
                               frames=200, interval=800, blit=true)

# Save to video file
anim["save"]("HMM.mp4", fps=15)
