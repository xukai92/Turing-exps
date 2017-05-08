# Load packages for inference
using Distributions
using Turing

# Load toy dataset
include("topic.data.jl");

# Define the Bayesian Mixture of Categorical (BayesMoC) model as
#
#            θ ~ Dir(α)
#           ϕₖ ~ Dir(β)
#         zₘ|Θ ~ Cat(θ)
#     Wₙₘ|zₘ,β ~ Cat(β_zₘ)
#
# with parameters below
#
#   K   - topic num
#   V   - vocabulary
#   M   - doc num
#   N   - total number of words models more similar
#   w   - word instances
#   doc - doc instances
#   β   - topic prior
#   α   - word prior
@model BayesMoC(K, V, M, N, w, doc, β, α) = begin
  θ ~ Dirichlet(α)

  ϕ = Vector{Vector{Real}}(K)
  for k = 1:K
    ϕ[k] ~ Dirichlet(β)
  end

  z = tzeros(Int, N)    # Turing-safe array
  for m = 1:M
    z[m] ~ Categorical(θ)
  end

  for n = 1:N
    w[n] ~ Categorical(ϕ[z[doc[n]]])
  end
end

# Collect 1000 samples using NUTS
samples = sample(BayesMoC(data=topicdata), Gibbs(250, PG(50, 1, :z), HMCDA(100, 0.1, 0.3, :θ, :ϕ)))

# Save result for vis
include("topic.helper.jl")
ldaresult = samples2visdata(samples)
open("/home/kai/projects/Turing-exps/amazon-talk/BayesMoC.result.json", "w") do f
    JSON.print(f, ldaresult)
end
