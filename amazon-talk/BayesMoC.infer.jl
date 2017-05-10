using Distributions, Turing   # load packages
include("topic.data.jl")      # load toy dataset

# Define the Bayesian Mixture of Categorical (BayesMoC) model
#   K   - topic num         ||
#   V   - vocabulary        ||             Model
#   M   - doc num           ||
#   N   - number of words   ||            θ ~ Dir(α)
#   w   - word instances    ||           ϕₖ ~ Dir(β)
#   doc - doc instances     ||         zₘ|Θ ~ Cat(θ)
#   β   - topic prior       ||     Wₙₘ|zₘ,β ~ Cat(β_zₘ)
#   α   - word prior        ||
@model BayesMoC(K, V, M, N, w, doc, β, α) = begin
  θ ~ Dirichlet(α)

  ϕ = Vector{Vector{Real}}(K)
  for k = 1:K
    ϕ[k] ~ Dirichlet(β)
  end

  z = tzeros(Int, M)    # Turing-safe array
  for n = 1:N
    if z[doc[n]] == 0
      z[doc[n]] ~ Categorical(θ)
    end
    w[n] ~ Categorical(ϕ[z[doc[n]]])
  end
end

# Collect 1000 samples using a compositional Gibbs sampler which combines
# - Hamiltonian Monte Carlo with Dual Averaging for θ and ϕ
#   * step-size 0.1 and length 0.3 for HMCDA; 1 iteration in each Gibbs
# - Particle Gibbs for discrete variable s
#   * 50 particles are used for PG; 1 iteration in each Gibbs
samples = sample(
  BayesMoC(data=topicdata),
  Gibbs(1000, PG(50, 1, :z), HMCDA(200, 0.65, 1.5, :θ, :ϕ))
)


#####################################
# Below are codes for visualization #
#####################################

# Save result for vis
include("topic.helper.jl")

# ldaresult = samples2visdata(samples)
# open("/home/kai/projects/Turing-exps/amazon-talk/BayesMoC.result.json", "w") do f
#     JSON.print(f, ldaresult)
# end

for i = 1:length(samples[:ϕ])
  makerectbinplot(samples, i, "frames/BayesMoC")
end
