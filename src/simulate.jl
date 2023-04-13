function simulate(T::Integer, β::FLOATMAT, Σ::PDMat)
    @assert T > 0 "The number of observations must be strictly positive."
    
    d, p = size(β)
    d = Integer((d - 1)/p)

    str = "β is a $(size(β, 1)) x $(size(β, 2)) matrix,\nwhile Σ is $(size(Σ, 1)) x $(size(Σ, 2)), \nS is $(size(S, 1)) x $(size(S, 2))."
    !(p == size(Σ, 1) == size(Σ, 2)) && throw(DimensionMismatch(str))

    # Define the error distribution
    error_dist = MvNormal(Σ)

    # Storage
    F = ones(1+d*p)
    y = zeros(T, p)
    for t = 1:T
        if t <= d
            y[t, :] = rand(error_dist)
        else
            if d > 0
                F[2:end] = vec(y[t-d:t-1, :])
            end
            y[t, :] = F' * β .+ rand(error_dist)'
        end
    end

    return y
end