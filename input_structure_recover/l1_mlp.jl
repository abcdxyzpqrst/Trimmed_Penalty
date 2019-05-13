include("solvers.jl")
using Distributions, NPZ
using Random
#-----------------------------------
#       Control Panel
#-----------------------------------
Random.seed!(12345678)
p = 80
h = 20
s = 4
n_list = [2000]
n_test = 1500
max_sims = 1
α = (p-s)/p
λ = 0.1
ξ = 0.05

Wᵀ = zeros(h, p)    # true weight
for i = 1:h
    for j = (i-1)*s+1:i*s
        Wᵀ[i, j] = sqrt(p/(h*s)) * randn()
    end
end
o  = ones(h, 1)

#npzwrite("synthetic_true.npy", Wᵀ)

# data loading
μ = zeros(p)
Σ = eye(p)
d = MvNormal(μ, Σ)

train_loss = zeros(length(n_list))
test_loss = zeros(length(n_list))
corr = zeros(length(n_list))
l1_norm = zeros(length(n_list))

#Z = sqrt(1/h) * randn(h, p)
#W₀ = Wᵀ + Z
W₀ = sqrt(1/h) * randn(h, p)
τ = sum(abs.(Wᵀ))
trim = floor(Int64, α*h*p)
w = (trim/(h*p)) * ones(h, p)

for i = 1:length(n_list)
    n = n_list[i]
    for sim = 1:max_sims
        X = rand(d, n); X = X';
        y = (o'*ReLU.(Wᵀ*X'))'

        X_test = rand(d, n_test); X_test = X_test';
        y_test = (o'*ReLU.(Wᵀ*X_test'))'

        train_ℓ₁_mlp(W₀, o, X, y, τ; itm=10000)
        train_loss[i] += empirical_loss(W₀, o, X, y)
        test_loss[i] += empirical_loss(W₀, o, X_test, y_test)
        corr[i] += correlation(W₀, Wᵀ)
        l1_norm[i] += sum(abs.(W₀)) / sum(abs.(Wᵀ))
    end
    println(n, " samples finished!")
end
train_loss /= max_sims
test_loss /= max_sims
corr /= max_sims
l1_norm /= max_sims

println(train_loss)
println(test_loss)
println(corr)
println(l1_norm)

W_perm = permuted_weight(W₀, Wᵀ)
npzwrite("synthetic_l1_random_1000.npy", W_perm)
