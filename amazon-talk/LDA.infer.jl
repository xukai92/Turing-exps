# Load packages for inference
using Distributions
using Turing

# Load toy dataset
include("topic.data.jl");

# Define the LDA model with parameters:
#   K   - topic num
#   V   - vocabulary
#   M   - doc num
#   N   - total number of words
#   w   - word instances
#   doc - doc instances
#   β   - topic prior
#   α   - word prior
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
    z[n] ~ Categorical(θ[doc[n]])
  end

  for n = 1:N
    w[n] ~ Categorical(ϕ[z[doc[n]]])
  end
end

# Collect 1000 samples using NUTS
samples = sample(LDA(data=ldadata), NUTS(250, 0.65))

# Samples can be got and used like below
ϕs = samples[:ϕ][1:200]   # fetch first 200 samples for ϕ

# Usually the mean of samples are used learning result, e.g. ϕ = Eᵢ[ϕᵢ]
ϕ = mean(samples[:ϕ])

# Save result for vis
include("topic.helper.jl")
ldaresult = samples2visdata(samples)
open("/home/kai/projects/Turing-exps/amazon-talk/LDA.result.json", "w") do f
    JSON.print(f, ldaresult)
end
