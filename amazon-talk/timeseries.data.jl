K = 5
N = 21
initial = fill(1.0 / K, K)
m = (collect(1.0:K)*2-K)*2

# Define transition matrix
T = zeros(K,K); for i=1:K; T[i,:] = rand(Dirichlet(ones(K)./K)); end
T = T + K*eye(K)/K; for i=1:K; T[i,:] = T[i,:] ./ sum(T[i,:]); end # Add self-trans prob.


@model hmmdata(initial, T, m) = begin
    states = tzeros(Int,N)
    # T = TArray{Array{Float64,}}
    y = zeros(N)

    states[1] ~ Categorical(initial)
    y[1] ~ Normal(m[states[1]], 0.4)
    for i = 2:N
        states[i] ~ Categorical(vec(T[states[i-1],:]))
        y[i] ~ Normal(m[states[i]], 0.4)
    end

end

srand(1234)
chain = sample(hmmdata(initial, T, m), PG(10,2));

y = chain[:y][1]
srand()
