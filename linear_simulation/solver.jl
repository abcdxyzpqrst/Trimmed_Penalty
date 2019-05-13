##### solver functions #####
using Roots
using LinearAlgebra
"""
Parameter Configurations
X:  observations
y:  responses
θ:  variables
w:  weight-parameter for θ (for trimmed penalty)
λ:  coefficient of regularizer
ρ:  parameter for nonconvex penalties, SCAD and MCP
κ:  proximal parameters (κ = η*λ)
τ:  stepsize for w
α:  belief of sparsity level
h:  trimming parameter
"""

# Just utils
function eye(p)
    I = zeros(p, p)
    for i = 1:p
        I[i, i] = 1.0
    end
    return I
end

# Since the objective is "sum", the corresponding λ value should be scaled
function trim_objective(X, y, θ, w, λ)
    return 0.5*sum(abs2, X*θ - y) + λ*sum(abs, θ.*w)
end

function scad_objective(X, y, θ, ρ, λ)
    return 0.5*sum(abs2, X*θ - y) + λ*scad(θ, λ, ρ)
end

function mcp_objective(X, y, θ, ρ, λ)
    return 0.5*sum(abs2, X*θ - y) + λ*mcp(θ, λ, ρ)
end

function vanilla_objective(X, y, θ, λ)
    return 0.5*sum(abs2, X*θ - y) + λ*sum(abs, θ)
end

function scad(θ, λ, ρ)
    val = 0.0
    for i in eachindex(θ)
        abs(θ[i]) ≤ λ ? val += λ*abs(θ[i]) :
        abs(θ[i]) > ρ*λ ? val += 0.5*(ρ+1)*λ^2 :
        val += (2*ρ*λ*abs(θ[i]) - λ^2 - θ[i]^2) / (2*(ρ-1))
    end
    return val
end

function mcp(θ, λ, ρ)
    val = 0.0
    for i in eachindex(θ)
        abs(θ[i]) ≤ ρ*λ ? val += λ*abs(θ[i]) - θ[i]^2/(2*ρ) :
        val += 0.5*ρ*λ^2
    end
    return val
end

function prox_ℓ₀_norm(θ, κ)
    ν = sqrt(2.0*κ)
    for i in eachindex(θ)
        abs(θ[i]) ≤ ν && (θ[i] == 0.0);
    end
end

function prox_ℓ₁_norm(θ, κ)
    for i in eachindex(θ)
        abs(θ[i]) > κ ? θ[i] = sign(θ[i])*max(abs(θ[i])-κ, 0) :
        θ[i] = 0.0
    end
end

function prox_trimmed_ℓ₁_norm(θ, w, κ)
    for i in eachindex(θ)
        ν = κ*w[i]
        abs(θ[i]) > ν ? θ[i] = sign(θ[i])*max(abs(θ[i])-ν, 0) :
        θ[i] = 0.0
    end
end

function prox_ℓ₁₂_norm(θ, w, κ)
    for i = 1:length(w)
        ν = κ*w[i]
        ρ = vecnorm(x[i, :]); r = ρ;
        ρ == 0.0 && continue;
        ρ > ν ? r -= ν : r = 0.0;
        x[i, :] = (r/ρ) * x[i, :]
    end
end

function prox_capped_simplex(w, lb, ub, h)
    h == length(w) && (fill!(w, 1.0); return nothing;)
    h  > length(w) && error("wrong h or size of vector!");

    h = Float64(h)
    f(α) = sum(max.(min.(w .- α, ub), lb)) - h
    a = -1.5 + minimum(w)
    b = maximum(w)
    α = fzero(f, [a, b])
    for i in eachindex(w)
        w[i] -= α
        w[i] > ub ? w[i] = ub :
        w[i] < lb ? w[i] = lb :
        nothing
    end
end

function prox_scad(θ, ρ, κ)
    for i in eachindex(θ)
        abs(θ[i]) < 2*κ ? θ[i] = sign(θ[i])*max(abs(θ[i])-κ, 0) :
        abs(θ[i]) > ρ*κ ? θ[i] = θ[i] :
        θ[i] = ((ρ-1)*θ[i] - sign(θ[i])*ρ*κ)/(ρ-2)
    end
end

function prox_mcp(θ, ρ, κ)
    for i in eachindex(θ)
        abs(θ[i]) > ρ*κ ? θ[i] = θ[i] :
        abs(θ[i]) ≤ κ ? θ[i] = 0.0 :
        θ[i] = (ρ*θ[i] - sign(θ[i])*ρ*κ)/(ρ-1)
    end
end

# From here, we use vanilla gradient descent, ∇θ = ∂𝓛/∂θ, θ⁽ᵗ⁺¹⁾ = θ⁽ᵗ⁾ - η∇θ
function solve_trim_lasso(X, y, θ, w, h, λ; itm=1000, tol=1e-6, ptf=100)
    θ⁻ = copy(θ)
    η  = 1.0/opnorm(X)^2
    κ  = η*λ
    τ  = 1.0

    err = 1.0
    noi = 0

    while err ≥ tol
        # proximal gradient on θ
        ∇θ = X'*(X*θ - y)
        BLAS.axpy!(-η, ∇θ, θ)
        prox_trimmed_ℓ₁_norm(θ, w, κ)

        # projected gradient on w
        BLAS.axpy!(-τ, abs.(θ), w)
        prox_capped_simplex(w, 0.0, 1.0, h)
        
        # update informations
        obj = trim_objective(X, y, θ, w, λ)
        err = norm(θ⁻ - θ)/η
        copyto!(θ⁻, θ)

        # print learning information
        noi += 1
        #noi % ptf == 0 && @printf("iter %4d, obj %1.2e, err %1.2e\n",
        #                          noi, obj, err);
        noi ≥ itm && break;
    end
end

function solve_mcp_lasso(X, y, θ, ρ, λ; itm=1000, tol=1e-6, ptf=100)
    θ⁻ = copy(θ)
    η  = 1.0/opnorm(X)^2
    κ  = η*λ

    err = 1.0
    noi = 0

    while err ≥ tol
        ∇θ = X'*(X*θ - y)
        BLAS.axpy!(-η, ∇θ, θ)
        prox_mcp(θ, ρ, κ)

        obj = mcp_objective(X, y, θ, ρ, λ)
        err = norm(θ⁻ - θ)/η
        copyto!(θ⁻, θ)

        noi += 1
        #noi % ptf == 0 && @printf("iter %4d, obj %1.2e, err %1.2e\n",
        #                          noi, obj, err);
        noi ≥ itm && break;
    end
end

function solve_scad_lasso(X, y, θ, ρ, λ; itm=1000, tol=1e-6, ptf=100)
    θ⁻ = copy(θ)
    η  = 1.0/opnorm(X)^2
    κ  = η*λ

    err = 1.0
    noi = 0

    while err ≥ tol
        ∇θ = X'*(X*θ - y)
        BLAS.axpy!(-η, ∇θ, θ)
        prox_scad(θ, ρ, κ)

        obj = scad_objective(X, y, θ, ρ, λ)
        err = norm(θ⁻ - θ)/η
        copyto!(θ⁻, θ)

        noi += 1
        #noi % ptf == 0 && @printf("iter %4d, obj %1.2e, err %1.2e\n",
        #                          noi, obj, err);
        noi ≥ itm && break;
    end
end

function solve_vanilla_lasso(X, y, θ, λ; itm=1000, tol=1e-6, ptf=100)
    θ⁻ = copy(θ)
    η  = 1.0/opnorm(X)^2
    κ  = η*λ

    err = 1.0
    noi = 0
    
    while err ≥ tol
        ∇θ = X'*(X*θ - y)
        BLAS.axpy!(-η, ∇θ, θ)
        prox_ℓ₁_norm(θ, κ)

        obj = vanilla_objective(X, y, θ, λ)
        err = norm(θ⁻ - θ)/η
        copyto!(θ⁻, θ)

        noi += 1
        #noi % ptf == 0 && @printf("iter %4d, obj %1.2e, err %1.2e\n",
        #                          noi, obj, err);
        noi ≥ itm && break
    end
end
