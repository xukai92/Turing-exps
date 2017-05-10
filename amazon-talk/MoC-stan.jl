using Stan

include("topic.data2.jl")
const nbstandata = [topicdata]

const naivebayesstanmodel = "
// supervised naive Bayes
data {
  // training data
  int<lower=1> K;               // num topics
  int<lower=1> V;               // num words
  int<lower=0> M;               // num docs
  int<lower=0> N;               // total word instances
  int<lower=1,upper=K> z[M];    // topic for doc m
  int<lower=1,upper=V> w[N];    // word n
  int<lower=1,upper=M> doc[N];  // doc ID for word n
  // hyperparameters
  vector<lower=0>[K] alpha;     // topic prior
  vector<lower=0>[V] beta;      // word prior
}
parameters {
  simplex[K] theta;   // topic prevalence
  simplex[V] phi[K];  // word dist for topic k
}
model {
  // priors
  theta ~ dirichlet(alpha);
  for (k in 1:K)
    phi[k] ~ dirichlet(beta);
  // likelihood, including latent category
  for (m in 1:M)
    z[m] ~ categorical(theta);
  for (n in 1:N)
    w[n] ~ categorical(phi[z[doc[n]]]);
}
"

stan_model_name = "Naive_Bayes"
nbstan = Stanmodel(Sample(save_warmup=true), name=stan_model_name, model=naivebayesstanmodel, nchains=1);

nb_stan_sim = stan(nbstan, nbstandata, CmdStanDir=CMDSTAN_HOME, summary=false);
# nb_stan_sim.names
nb_stan_sim = Stan.read_stanfit_warmup_samples(nbstan)

stan_d_raw = Dict()
for i = 1:4, j = 1:10
  stan_d_raw["phi[$i][$j]"] = nb_stan_sim[1:1000, ["phi.$i.$j"], :].value[:]
end

stan_d = Dict()
for i = 1:4
  stan_d["phi[$i]"] = [[stan_d_raw["phi[$i][$k]"][j] for k = 1:10] for j = 1:1000]
end


K = nbstandata[1]["K"]
V = nbstandata[1]["V"]

using Gadfly
using DataFrames
Gadfly.push_theme(:dark)

makerectbinplot(i, fn) = begin

  ϕ = [mean(stan_d["phi[1]"][1:i])'; mean(stan_d["phi[2]"][1:i])'; mean(stan_d["phi[3]"][1:i])'; mean(stan_d["phi[4]"][1:i])']

  df = DataFrame(Topic = vec(repmat(collect(1:K)', V, 1)), Word = vec(repmat(collect(1:V)', 1, K)), Probability = vec(ϕ))

  p = plot(df,x=:Word, y=:Topic, color=:Probability, Geom.rectbin)

  draw(PNG("$fn$i.png", 6inch, 4.5inch), p)
end

for i = 1:length(stan_d["phi[1]"])
  makerectbinplot(i, "frames/MoC-stan")
end
