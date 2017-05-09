# Load packages
using Distributions, Turing

# Load toy dataset
include("timeseries.data.jl")

# Define a Bayesian Hidden Markov Model model with parameters:
#   N   -   number of observations
#   K   -   number of states
#   y   -   observations
@model BayesHMM(N, K, y) = begin
    s = tzeros(Int, N); m = tzeros(Real, K); T = Vector{Vector{Real}}(K)
    for i = 1:K
        T[i] ~ Dirichlet(ones(K)/K)
        m[i] ~ Normal(i, 1)
    end
    s[1] ~ Categorical(ones(Float64, K)/K)
    for i = 2:N
        s[i] ~ Categorical(vec(T[s[i-1]]))
        y[i] ~ Normal(m[s[i]], 1)
    end
end

# Collect 500 samples using a compositional Gibbs sampler which combines
# - Hamiltonian Monte Carlo for continuous variables m and T
#   * step-size 0.2 and step-length 5 for HMC; 1 iteration in each Gibbs
# - Particle Gibbs for discrete variable s
#   * 50 particles are used for PG; 1 iteration in each Gibbs
samples = sample(
  BayesHMM(N, K, y),
  Gibbs(300, HMC(1, 0.2, 5, :m, :T), PG(50, 1, :s))
)



#####################################
# Below are codes for visualization #
#####################################

using PyPlot, PyCall
@pyimport matplotlib.animation as animation
plt["style"]["use"]("ggplot")

fig, ax = plt[:subplots]();
ax[:set_xlim](( 0, N))
ax[:set_ylim]((-20, 20))
line, = ax[:plot]([], [], lw=2)

# initialization function: plot the background of each frame
function init()
    ax[:plot](1:N, y, "rs")
    return (line,)
end

# animation function. This is called sequentially
function animate(i)
    n = mod(i,length(samples.value))
    line[:set_data](1:N, samples[:m][n+1][samples[:s][n+1]])
    return (line,)
end

# call the animator. blit=True means only re-draw the parts that have changed.
anim = animation.FuncAnimation(fig, animate, init_func=init,
                               frames=200, interval=800, blit=true)

anim["save"]("BayesHMM.mp4", fps=15)
