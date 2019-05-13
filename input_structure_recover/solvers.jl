using Roots
using LinearAlgebra

function eye(p)
    I = zeros(p,p)
    for i = 1:p
        I[i,i] = 1.0
    end
    return I
end

function ReLU(x)
    return max.(0, x)
end

function dReLU(x)
    if x > 0.0
        return 1.0
    else
        return 0.0
    end
end

function gradient(W, o, X, y)
    N = size(X, 1)

    # forward
    z = W*X'
    a = ReLU.(z)
    ŷ = o'*a
    
    # backward
    δ = ŷ' .- y
    err = (o .* δ') .* dReLU.(z)
    W_grad = err * X / N

    return W_grad
end

function objective(W, o, X, y)
    """
    model:  y = oᵀσ(W*Xᵀ) where σ(⋅) is an activation function
    """
    ŷ = o'*ReLU.(W*X')
    return 0.5*mean(abs2.(y .- ŷ')) 
end

function empirical_loss(W, o, X, y)
    """
    empirical loss is defined as var[y - ŷ]/var[y]
    """
    ŷ = o'*ReLU.(W*X')
    numerator = var(y .- ŷ')
    denominator = var(y)
    return numerator / denominator
end

function maximum_row(rᵀ, Ŵ)
    """
    rᵀ: the row of true weight
    Ŵ : the esimated weight
    """
    h = size(Ŵ, 1)
    largest = -Inf
    largest_index = 1
    for j = 1:h
        r̂ = Ŵ[j, :]
        numerator = rᵀ'*r̂
        denominator = norm(rᵀ)*norm(r̂)
        if numerator/denominator > largest
            largest = numerator/denominator
            largest_index = j
        end
    end
    return largest, largest_index
end

function correlation(Ŵ, Wᵀ)
    """
    correlation with true weight Wᵀ
    """
    h = size(Wᵀ, 1)
    corr = 0.0
    for i = 1:h
        corr += maximum_row(Wᵀ[i, :], Ŵ)[1]
    end
    return corr / h
end

function permuted_weight(Ŵ, Wᵀ)
    h, p = size(Wᵀ)
    permuted = zeros(h,p)
    processed = []
    not_processed = []
    id_list = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
    for i = 1:h
        largest_index = maximum_row(Wᵀ[i, :], Ŵ)[2]
        permuted[i, :] = Ŵ[largest_index, :]
        if largest_index in processed
            push!(not_processed, largest_index)
        else
            push!(processed, largest_index)
        end
    end
    return permuted
end

function proj_ℓ₁_ball(W, τ)
    """
    Projection W onto the ℓ₁ ball scaled by the ℓ₁-ball of radius τ
    """
    f(α) = sum(max.(abs.(W) .- α, 0.0)) - τ
    m = -1.0
    M = maximum(abs.(W))
    r = fzero(f, [m, M]) 
    for i in eachindex(W)
        W[i] = sign(W[i]) * max(abs(W[i]) - r, 0.0)
    end
end

function proj_ℓ₀_ball(W, sh)
    """
    Projection W onto the set of sh sparse matrices
    z:  number of zeros
    V:  vectorized W
    I:  indices for smallest z entries
    """
    z = length(W) - sh
    V = vec(W)
    I = sortperm(abs.(V))[1:z]
    for i in I
        V[i] = 0.0
    end
end

function prox_trimmed_ℓ₁_norm(W, w, κ)
    for i in eachindex(W)
        ν = κ*w[i]
        abs(W[i]) > ν ? W[i] = sign(W[i])*max(abs(W[i]) - ν, 0) :
        W[i] = 0.0
    end
end

function proj_capped_simplex(w, lb, ub, h)
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

function train_naive_mlp(W, o, X, y; itm=2000, ptf=100)
    η = 0.1
    for epoch = 1:itm
        W_grad = gradient(W, o, X, y)
        BLAS.axpy!(-η, W_grad, W)

        obj = objective(W, o, X, y)
        #epoch % ptf == 0 && println("Loss:", obj)
    end
end

function train_ℓ₁_mlp(W, o, X, y, τ; itm=2000, ptf=100)
    η = 0.1
    for epoch = 1:itm
        W_grad = gradient(W, o, X, y)
        BLAS.axpy!(-η, W_grad, W)

        proj_ℓ₁_ball(W, τ)
        obj = objective(W, o, X, y)
        #epoch % ptf == 0 && println("Loss:", obj)
    end
end

function train_ℓ₀_mlp(W, o, X, y, sh; itm=100000, ptf=10000)
    η = 0.1
    for epoch = 1:itm
        W_grad = gradient(W, o, X, y)
        BLAS.axpy!(-η, W_grad, W)
        
        proj_ℓ₀_ball(W, sh)
        obj = objective(W, o, X, y)
        #epoch % ptf == 0 && println("Loss:", obj)
    end
end

function train_trim_mlp(W, o, X, y, w, h; η=0.1, λ=0.1, ξ=0.075, itm=100000, ptf=100)
    κ = η*λ
    for epoch = 1:itm
        W_grad = gradient(W, o, X, y)
        BLAS.axpy!(-η, W_grad, W)
        prox_trimmed_ℓ₁_norm(W, w, κ)
        
        BLAS.axpy!(-ξ, abs.(W), w)
        proj_capped_simplex(w, 0.0, 1.0, h)
    end
end
