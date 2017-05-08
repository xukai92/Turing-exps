using Distributions
using Turing

y = [ 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0, 2.0, 2.0, 2.0, 1.0, 1.0 ];
N = length(y);  K = 3;

@model BayesHmm(y) = begin
    s = tzeros(Int64, N)
    m = tzeros(Real, K)
    T = Array{Array}(K)
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
c = sample(BayesHmm(y), g);

m = c[:m][111];
s = c[:s][111];

using Gadfly

p_layer = layer(x=1:N, y=y, Geom.point, Theme(default_color=colorant"royalblue"))
l_layer = layer(x=1:N, y=m[s], Geom.line)

plt = plot(p_layer, l_layer);

draw(PNG("amazon-talk/BayesHMM.png", 6inch, 4.5inch), plt)
