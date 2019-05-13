##### main scripts #####
include("solver.jl")
using Distributions 
using Random
using ArgParse
using LinearAlgebra
using PyCall
@pyimport matplotlib.pyplot as plt
#-------------------------------------------------------------------
#   argument parsing
#-------------------------------------------------------------------
s = ArgParseSettings()
s.description = "log ℓ₂-error comparison"
s.exc_handler = ArgParse.debug_handler
@add_arg_table s begin
    ("--type"; arg_type=String; default="incoherent"; help="type of data matrix")
    ("--itm"; arg_type=Int; default=10000; help="maximum iterations")
    ("--lam"; arg_type=Float64; default=16.0; help="reg. params")
end
args = ARGS
isa(args, AbstractString) && (args=split(args))
if in("--help", args) || in("-h", args)
    ArgParse.show_help(s; exit_when_done=false)
    return
end
o = parse_args(args, s; as_symbols=true)
println(o[:type])
##### parameter configuration #####
Random.seed!(1)         # random seed
n = 160                 # number of observations
p = 256                 # problem dimension
k = 16                  # sparsity index
σ = 1.0                 # noise level
α = (p-k)/p             # belief of sparsity level
λ = 16.0                # coefficient of regularizer
a = 3.0                 # SCAD parameter
b = 2.5                 # MCP parameter

##### normal distribution #####
typ = o[:type]
μ  = zeros(p)
if typ == "nonincoherent"
    δ = 2.0/k
    M₁ = eye(p)
    M₁[1:k, k+1] = δ*ones(k,1)
    M₁[k+1, 1:k] = δ*ones(1,k)
    d = MvNormal(μ, M₁)
elseif typ == "incoherent"
    δ = 0.7
    M₂ = δ*ones(p, p) + (1-δ)*eye(p)
    d  = MvNormal(μ, M₂)
else
    error("Type error!")
end

##### data loading #####
X  = rand(d, n); X = X';    # covariates (design matrix)
θᵀ = zeros(p)
id = randperm(p)[1:k]
for i in id
    θᵀ[i] = 4.0*randn()
end
y  = X*θᵀ + σ*randn(n)      # responses

##### gradient descent parameter #####
itm = o[:itm]
tol = 1e-6
ptf = 1000
sim = 50

min_values = []
x = collect(1:1:itm)
plt.rc("text", usetex=true)
plt.rc("font", family="Times New Roman", size=28)
plt.figure()
plt.xlabel("Iteration Count, \$t\$")
plt.ylabel("\$\\log(\\Vert\\beta^t - \\beta^*\\Vert_2)\$")
plt.xlim([0, itm]) 
##### Trimmed Lasso #####
for i = 1:sim
    trim_error = []    
    θ = 5.0*randn(p)
    h = floor(Int64, α*p)
    w = fill(h/p, p)

    for iter = 1:itm
        solve_trim_lasso(X, y, θ, w, h, λ; itm=1, tol=tol, ptf=ptf)
        push!(trim_error, log(norm(θᵀ .- θ)))
    end
    plt.plot(x, trim_error, color="#1f77b4", linewidth=0.5)
end
#plt.xticks([0, 500, 1000])
plt.legend(["\$\\log\\ell_2\$-error"])
file_name = string("stationary_", typ, ".pdf")
plt.savefig(file_name, bbox_inches="tight", transparent=true)
println("success")
#plt.legend(loc="best")
