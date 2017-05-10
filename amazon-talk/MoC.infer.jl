using Distributions, Turing   # load packages
include("topic.data1.jl")     # load toy dataset

# Define the Mixture of Categorical (MoC) model
#   K   - topic num         ||
#   V   - vocabulary        ||             Model
#   M   - doc num           ||
#   N   - number of words   ||            θ ~ Dir(α)
#   z   - doc topic idx     ||           ϕₖ ~ Dir(β)
#   w   - word instances    ||         zₘ|Θ ~ Cat(θ)
#   doc - doc instances     ||     Wₙₘ|zₘ,β ~ Cat(β_zₘ)
#   β   - topic prior       ||
#   α   - word prior        ||
@model MoC(K, V, M, N, z, w, doc, β, α) = begin
  θ ~ Dirichlet(α)

  ϕ = Vector{Vector{Real}}(K)
  for k = 1:K
    ϕ[k] ~ Dirichlet(β)
  end

  for m = 1:M
    z[m] ~ Categorical(θ)   # z here is provided
  end

  for n = 1:N
    w[n] ~ Categorical(ϕ[z[doc[n]]])
  end
end

# Collect 1000 samples using the No-U-Turn sampler (NUTS)
samples = sample(MoC(data=topicdata), NUTS(1000, 200, 0.65))


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
  makerectbinplot(i, "frames/MoC")
end
