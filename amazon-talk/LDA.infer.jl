using Distributions, Turing   # load packages
include("topic.data3.jl")      # load toy dataset

# Define the LDA model with parameters:
#   K   - topic num         w   - word instances
#   V   - vocabulary      doc   - doc instances
#   M   - doc num           β   - topic prior
#   N   - count of words    α   - word prior
@model LDA(K, V, M, N, w, doc, β, α) = begin
  θ = Vector{Vector{Real}}(M)
  for m = 1:M
    θ[m] ~ Dirichlet(α)
  end

  ϕ = Vector{Vector{Real}}(K)
  for k = 1:K
    ϕ[k] ~ Dirichlet(β)
  end

  z = tzeros(Int, N)    # Turing-safe array
  for n = 1:N
    z[n] ~ Categorical(θ[doc[n]])   # z here is unkown
  end

  for n = 1:N
    w[n] ~ Categorical(ϕ[z[n]])
  end
end

# Collect 1000 samples using a compositional Gibbs sampler which combines
# - Particle Gibbs for discrete variable z
#   * 50 particles are used for PG; 1 iteration in each Gibbs
# - Hamiltonian Monte Carlo with Dual Averaging for θ and ϕ
#   * adaptation-step-num 200, target-accept-rate 0.65 and length 1.5 for HMCDA; 1 iteration in each Gibbs
samples = sample(
  LDA(data=topicdata),
  Gibbs(1000, PG(50, 1, :z), HMCDA(200, 0.65, 1.5, :θ, :ϕ))
)



#####################################
# Below are codes for visualization #
#####################################

K = topicdata["K"]
V = topicdata["V"]

using Gadfly
using DataFrames
Gadfly.push_theme(:dark)

makerectbinplot(i, fn) = begin

  ϕarr = mean(samples[:ϕ][1:i])
  ϕ = [ϕarr[1]'; ϕarr[2]'; ϕarr[3]'; ϕarr[4]']


  df = DataFrame(Topic = vec(repmat(collect(1:K)', V, 1)), Word = vec(repmat(collect(1:V)', 1, K)), Probability = vec(ϕ))

  p = plot(df,x=:Word, y=:Topic, color=:Probability, Geom.rectbin)

  draw(PNG("$fn$i.png", 6inch, 4.5inch), p)
end

for i = 1:length(samples[:ϕ])
  makerectbinplot(i, "frames/LDA")
end
