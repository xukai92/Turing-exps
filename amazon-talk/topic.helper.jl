# Load package for outputing results for visualization
using JSON

samples2visdata(samples) = begin
  # Convert ϕ from array-of-array to matrix
  ϕarr = mean(samples[:ϕ])
  ϕ = [ϕarr[1]'; ϕarr[2]']

  # Convert θ from array-of-array to matrix
  θarr = mean(samples[:θ])
  if length(θarr) == topicdata["M"]
    θ = reduce((a, b) -> cat(1, a, b'), Matrix{Float64}(0, topicdata["K"]), θarr)
  else
    θ = repmat(θarr', topicdata["M"], 1)
  end

  # Build a vector storing lengths of docs for vis
  doclist = topicdata["doc"]
  docldict = reduce((a, b) -> if haskey(a, b) a[b] += 1; a else a[b] = 1; a end, Dict(), doclist)
  docls = map(i -> docldict[i], 1:topicdata["M"])

  # Build a vector of word frequencies for vis
  wordlist = topicdata["w"]
  freqdict = reduce((a, b) -> if haskey(a, b) a[b] += 1; a else a[b] = 1; a end, Dict(), wordlist)
  freq = map(i -> freqdict[i], 1:topicdata["V"])

  # Save result for vis
  ldaresult = Dict(
      "topic_term_dists" => ϕ',
      "doc_topic_dist" => θ',
      "doc_lengths" => docls,
      "vocab" => 1:topicdata["V"],
      "term_frequency" => freq
  )
end

using Gadfly
using DataFrames
Gadfly.push_theme(:dark)

makerectbinplot(samples, i, fn) = begin
  K = topicdata["K"]
  V = topicdata["V"]

  ϕarr = samples[:ϕ][i]
  ϕ = [ϕarr[1]'; ϕarr[2]']

  df = DataFrame(Topic = vec(repmat(collect(1:K)', V, 1)), Word = vec(repmat(collect(1:V)', 1, K)), Probability = vec(ϕ))

  p = plot(df,x=:Word, y=:Topic, color=:Probability, Geom.rectbin)

  draw(PNG("$fn$i.png", 6inch, 4.5inch), p)
end
