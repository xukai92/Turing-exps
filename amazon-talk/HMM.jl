using Distributions
using Turing

include("timeseries.data.jl")

initial = fill(1.0 / K, K)

T = zeros(K,K); for i=1:K; T[i,:] = rand(Dirichlet(ones(K)./K)); end
T = T + K*eye(K)/K; for i=1:K; T[i,:] = T[i,:] ./ sum(T[i,:]); end # Add self-trans prob.

m = (collect(1.0:K)*2-K)*2

@model HMM(N, K, y, T, m) = begin
    s = tzeros(Int64, N)
    s[1] ~ Categorical(ones(Float64, K) / K)
    for i = 2:N
        s[i] ~ Categorical(vec(T[s[i-1],:]))
        y[i] ~ Normal(m[s[i]], 0.01)
    end
end

samples = sample(HMM(N, K, y, T, m), PG(50, 1000));

# m = samples[:m][111];
s = mean(samples[:s]);

include("timeseries.vis.jl")

plottimeseries(y, s, "HMM")
