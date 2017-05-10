using Distributions
using Turing
using Stan

const ldastanmodel = "
data {
  int<lower=2> K;               // num topics
  int<lower=2> V;               // num words
  int<lower=1> M;               // num docs
  int<lower=1> N;               // total word instances
  int<lower=1,upper=V> w[N];    // word n
  int<lower=1,upper=M> doc[N];  // doc ID for word n
  vector<lower=0>[K] alpha;     // topic prior
  vector<lower=0>[V] beta;      // word prior
}
parameters {
  simplex[K] theta[M];   // topic dist for doc m
  simplex[V] phi[K];     // word dist for topic k
}
model {
  for (m in 1:M)
    theta[m] ~ dirichlet(alpha);  // prior
  for (k in 1:K)
    phi[k] ~ dirichlet(beta);     // prior
  for (n in 1:N) {
    real gamma[K];
    for (k in 1:K)
      gamma[k] <- log(theta[doc[n],k]) + log(phi[k,w[n]]);
    increment_log_prob(log_sum_exp(gamma));  // likelihood
  }
}
"

const ldastandata = [
Dict(
  "K" => 2,
  "V" => 5,
  "M" => 25,
  "N" => 262,
  "w" => [4, 3, 5, 4, 3, 3, 3, 3, 3, 4, 5, 3, 4, 4, 5,
3, 4, 4, 4, 3, 5, 4, 5, 2, 3, 3, 1, 5, 5, 1, 4,
3, 1, 2, 5, 4, 4, 3, 5, 4, 2, 4, 5, 3, 4, 1, 4,
4, 3, 2, 1, 2, 1, 2, 2, 2, 1, 2, 2, 3, 1, 2, 2,
4, 4, 5, 4, 5, 5, 4, 3, 5, 4, 4, 4, 2, 2, 1, 1,
2, 1, 3, 1, 2, 1, 1, 1, 3, 2, 3, 3, 5, 4, 5, 4,
3, 5, 4, 2, 2, 2, 1, 3, 2, 1, 3, 1, 3, 1, 1, 2,
1, 2, 2, 4, 4, 4, 5, 5, 4, 4, 5, 4, 3, 3, 3, 1,
3, 3, 4, 2, 1, 3, 4, 4, 5, 4, 4, 4, 3, 4, 3, 4,
5, 1, 2, 1, 3, 2, 1, 1, 2, 3, 3, 3, 3, 4, 1, 4,
4, 4, 4, 3, 4, 4, 1, 2, 2, 3, 3, 1, 1, 4, 1, 3,
1, 5, 3, 2, 2, 1, 1, 2, 3, 3, 4, 4, 5, 3, 4, 3,
1, 5, 5, 5, 3, 3, 4, 5, 3, 3, 3, 2, 3, 1, 3, 3,
1, 3, 1, 5, 5, 5, 2, 2, 3, 3, 3, 1, 1, 5, 5, 5,
3, 1, 5, 4, 1, 3, 3, 3, 3, 4, 2, 5, 1, 3, 5, 2,
5, 5, 2, 1, 3, 3, 5, 3, 5, 3, 3, 5, 1, 2, 2, 1,
1, 2, 1, 2, 3, 1, 1],
  "doc" => [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2,
2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4,
4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 8, 8, 8, 8,
8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 9, 9, 9, 9,
9, 9, 9, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10,
10, 10, 10, 10, 10, 10, 11, 11, 11, 11, 11, 11, 12,
12, 12, 12, 13, 13, 13, 13, 13, 13, 13, 13, 13, 14,
14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 15, 15, 15,
15, 15, 15, 15, 15, 15, 15, 15, 16, 16, 16, 16, 16,
16, 16, 16, 16, 16, 17, 17, 17, 17, 17, 17, 17, 17,
17, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 19, 19,
19, 19, 19, 19, 19, 19, 19, 20, 20, 20, 20, 20, 20,
20, 20, 20, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21,
22, 22, 22, 22, 22, 22, 22, 22, 23, 23, 23, 23, 23,
23, 23, 23, 23, 23, 24, 24, 24, 24, 24, 24, 24, 24,
24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24,
25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25],
  "alpha" => [0.5, 0.5],
  "beta" => [0.2, 0.2, 0.2, 0.2, 0.2]
)
]

stan_model_name = "LDA"
ldastan = Stanmodel(Sample(save_warmup=true), name=stan_model_name, model=ldastanmodel, nchains=1);

lda_stan_sim = stan(ldastan, ldastandata, CmdStanDir=CMDSTAN_HOME, summary=false);
# lda_stan_sim.names

lda_stan_d_raw = Dict()
for i = 1:2, j = 1:5
  lda_stan_d_raw["phi[$i][$j]"] = lda_stan_sim[1001:2000, ["phi.$i.$j"], :].value[:]
end

lda_stan_d = Dict()
for i = 1:2
  lda_stan_d["phi[$i]"] = [[lda_stan_d_raw["phi[$i][$k]"][j] for k = 1:5] for j = 1:1000]
end

K = ldastandata[1]["K"]
V = ldastandata[1]["V"]

using Gadfly
using DataFrames
Gadfly.push_theme(:dark)

makerectbinplot(i, fn) = begin

  ϕ = [mean(lda_stan_d["phi[1]"][1:i])'; mean(lda_stan_d["phi[2]"][1:i])']

  df = DataFrame(Topic = vec(repmat(collect(1:K)', V, 1)), Word = vec(repmat(collect(1:V)', 1, K)), Probability = vec(ϕ))

  p = plot(df,x=:Word, y=:Topic, color=:Probability, Geom.rectbin)

  draw(PNG("$fn$i.png", 6inch, 4.5inch), p)
end

for i = 1:length(lda_stan_d["phi[1]"])
  makerectbinplot(i, "frames/LDA-stan")
end
