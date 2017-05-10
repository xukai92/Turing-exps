using Distributions, Turing   # load packages
include("topic.data.jl")      # load toy dataset

# Define the LDA model with parameters:
#   K   - topic num         w   - word instances
#   V   - vocabulary      doc   - doc instances
#   M   - doc num           β   - word prior
#   N   - count of words    α   - topic prior
@model LDA(K, V, M, N, w, doc, β, α) = begin
  θ = Vector{Vector{Real}}(M)
  for m = 1:M
    θ[m] ~ Dirichlet(α)
  end

  ϕ = Vector{Vector{Real}}(K)
  for k = 1:K
    ϕ[k] ~ Dirichlet(β)
  end

  # z = tzeros(Int, N)    # Turing-safe array
  # for n = 1:N
  #   z[n] ~ Categorical(θ[doc[n]])   # z here is unkown
  # end

  for n = 1:N
    phi_dot_theta = [dot(map(p -> p[i], ϕ), θ[doc[n]]) for i = 1:V]
    w[n] ~ Categorical(phi_dot_theta)
  end
end

# Collect 1000 samples using a compositional Gibbs sampler which combines
# - Particle Gibbs for discrete variable z
#   * 50 particles are used for PG; 1 iteration in each Gibbs
# - Hamiltonian Monte Carlo with Dual Averaging for θ and ϕ
#   * adaptation-step-num 200, target-accept-rate 0.65 and length 1.5 for HMCDA; 1 iteration in each Gibbs
samples = sample(
  LDA(data=topicdata),
  HMCDA(1000, 0.65, 1.5)
  # Gibbs(1000, PG(50, 1, :z), HMCDA(200, 0.65, 1.5, :θ, :ϕ))
)



#####################################
# Below are codes for visualization #
#####################################

K = topicdata["K"]
V = topicdata["V"]
M = topicdata["M"]

using Gadfly
using DataFrames
Gadfly.push_theme(:dark)

makerectbinplot(i, fn) = begin

  θarr = mean(samples[:θ][1:i])
  θ = [θarr[1]'; θarr[2]';θarr[3]';θarr[4]';θarr[5]';θarr[6]';θarr[7]';θarr[8]';θarr[9]';θarr[10]';θarr[11]';θarr[12]';θarr[13]';θarr[14]';θarr[15]';θarr[16]';θarr[17]';θarr[18]';θarr[19]';θarr[20]';θarr[21]';θarr[22]';θarr[23]';θarr[24]';θarr[25]']


  df = DataFrame( Topic= vec(repmat(collect(1:K)', M, 1)), Doc = vec(repmat(collect(1:M)', 1, K)), Probability = vec(θ))

  p = plot(df,x=:Topic, y=:Doc, color=:Probability, Geom.rectbin)

  draw(PNG("$fn$i.png", 6inch, 4.5inch), p)
end

for i = 1:length(samples[:θ])
  makerectbinplot(i, "frames/LDA-theta")
end
