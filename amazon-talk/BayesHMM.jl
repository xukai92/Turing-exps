using Distributions
using Turing

include("timeseries.data.jl")

@model BayesHMM(N, K, y) = begin
    s = tzeros(Int64, N)
    m = tzeros(Real, K)
    T = Vector{Vector{Real}}(K)
    for i = 1:K
        T[i] ~ Dirichlet(ones(K)/K)
        # m[i] ~ Normal(1, 0.1) # Defining m this way causes label-switching problem.
        m[i] ~ Normal(i, 0.01)
    end
    s[1] ~ Categorical(ones(Float64, K)/K)
    for i = 2:N
        s[i] ~ Categorical(vec(T[s[i-1]]))
        y[i] ~ Normal(m[s[i]], 0.01)
    end
end

g = Gibbs(300, HMC(1, 0.2, 5, :m, :T), PG(50, 1, :s))
c = sample(BayesHMM(N, K, y), g);

ms = c[:m];
ss = c[:s];

yt = mean(map(i -> ms[i][ss[i]], 1:N))

include("timeseries.vis.jl")

plottimeseries(y, yt, "BayesHMM")
