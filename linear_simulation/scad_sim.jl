##### main scripts #####
include("solver.jl");
using Distributions, Random, ArgParse;
#------------------------------------------------------------------------
#   argument parsing
#------------------------------------------------------------------------
s = ArgParseSettings()
s.description = "SCAD Lasso Simulations"
s.exc_handler = ArgParse.debug_handler 
@add_arg_table s begin
    ("--n"; arg_type=Int; default=100; help="number of observations")
    ("--p"; arg_type=Int; default=128; help="problem dimension")
    ("--k"; arg_type=Int; default=8; help="sparsity index")
    ("--itm"; arg_type=Int; default=10000; help="maximum iterations")
    ("--type"; arg_type=String; default="inco"; help="type of data matrix")
end
args = ARGS
isa(args, AbstractString) && (args=split(args))
if in("--help", args) || in("-h", args)
    ArgParse.show_help(s; exit_when_done=false)
    return
end
o = parse_args(args, s; as_symbols=true)
println(o[:type])
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
if o[:type] == "nonincoherent"
    δ = 2.0/k
    M₁ = eye(p)
    M₁[1:k, k+1] = δ*ones(k,1)
    M₁[k+1, 1:k] = δ*ones(1,k)
    d = MvNormal(μ, M₁)
elseif o[:type] == "incoherent"
    δ = 0.7
    M₂ = eye(p)
    M₂ = δ*ones(p, p) + (1-δ)*eye(p)
    d = MvNormal(μ, M₂)
else
    error("Type error!")
end

##### simulations for SCAD penalty #####
println("\n##### SCAD Lasso Simulation for (n, p, k) = (", n, ", ", p, ", ", k, ") with ", o[:type], " matrix #####")
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
            
        solve_scad_lasso(X, y, θ, a, λ; itm=itm, tol=1e-6, ptf=1000)
        cnt += 1*(norm((sign.(abs.(θᵀ))) - (1*((abs.(θ)).>0.05)),1)==0)
    end
    suc[sim] = cnt
end
prob = norm((suc.>0),1)/max_sims
println(n, " samples finished, prob: ", prob)
