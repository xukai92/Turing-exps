# Load packages for inference
using Distributions
using Turing

# Load toy dataset
include("topic.data.jl");

# Define the Mixture of Categorical model with parameters:
#   K   - topic num
#   V   - vocabulary
#   M   - doc num
#   N   - total number of words
#   w   - word instances
#   doc - doc instances
#   β   - topic prior
#   α   - word prior
@model MoC(K, V, M, N, z, w, doc, α, β) = begin
  θ ~ Dirichlet(α)

  ϕ = Array{Real}(K)
  for k = 1:K
    ϕ[k] ~ Dirichlet(β)
  end

  for m = 1:M
    z[m] ~ Categorical(θ)
  end

  for n = 1:N
    w[n] ~ Categorical(ϕ[z[doc[n]]])
  end
end
