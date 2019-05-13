##### main scripts #####
include("solver.jl");
using Distributions, Random, ArgParse;
#------------------------------------------------------------------------
#   argument parsing
#------------------------------------------------------------------------
s = ArgParseSettings()
s.description = "Vanilla Lasso Simulations"
s.exc_handler = ArgParse.debug_handler 
@add_arg_table s begin
    ("--n"; arg_type=Int; default=100; help="number of observations")
    ("--p"; arg_type=Int; default=128; help="problem dimension")
    ("--k"; arg_type=Int; default=8; help="sparsity index")
    ("--itm"; arg_type=Int; default=10000; help="maximum iterations")
end
args = ARGS
isa(args, AbstractString) && (args=split(args))
if in("--help", args) || in("-h", args)
    ArgParse.show_help(s; exit_when_done=false)
    return
end
o = parse_args(args, s; as_symbols=true)
#------------------------------------------------------------------------
#   create data
#------------------------------------------------------------------------
##### parameter configuration #####
Random.seed!(12345678)      # random seed
n = o[:n]                   # number of observations
p = o[:p]                   # problem dimension
k = o[:k]                   # sparsity index
σ = 1.0                     # noise level
α = (p-k)/p                 # belief of sparsity level
λ = 50.0                    # coefficient of regularizer
a = 3.0                     # SCAD parameter
b = 2.5                     # MCP parameter
δ = 0.7                     # parameter for spiked-identity incoherent matrix
s = 5.0                     # true signal level

##### Gaussian distribution #####
μ  = zeros(p)
M₂ = δ*ones(p, p) + (1-δ)*eye(p)
d = MvNormal(μ, M₂)

##### simulations for vanilla ℓ₁ penalty #####
println("\n##### Vanilla Lasso Simulation for (n, p, k) = (", n, ", ", p, ", ", k, ") with incoherent matrix #####")
max_sims = 200                  # maximum number of simulations
itm = o[:itm]
suc = zeros(max_sims)
for sim = 1:max_sims
    cnt = 0
    X  = rand(d, n); X = X';    # covariates (design matrix)
    θᵀ = zeros(p)               # true variables
    id = randperm(p)[1:k]
    for i in id
        θᵀ[i] = s*randn()
    end
    y  = X*θᵀ + σ*randn(n)      # responses

    for λ in (10 .^ (collect(-3:0.2:1)))*n
        θ = zeros(p)
            
        solve_vanilla_lasso(X, y, θ, λ; itm=itm, tol=1e-6, ptf=1000)
        cnt += 1*(norm((sign.(abs.(θᵀ))) - (sign.(abs.(θ))), 1)==0)
    end
    suc[sim] = cnt
end
prob = norm((suc.>0),1)/max_sims
println(n, " samples finished, prob: ", prob)
