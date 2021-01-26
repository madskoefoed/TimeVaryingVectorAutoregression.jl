
"""
estimate(Priors)

Estimate a stochastic volatility model. Volatility is modelled as a random walk.

The hyperparameters 2/3 < β < 1 and 0 < δ ≤ 1 are discount factors, controlling the shocks to the Σ and Ω respectively.
"""

estimate(y::Vector{<:Real}, priors::Priors) = estimate(vec2mat(y), priors)

function estimate(y::Matrix{<:Real}, priors::Priors)
    m = priors.m
    P = priors.P
    S = priors.S
    R = P / priors.δ
    Ω = kron(S, R)

    T, p = size(y)
    d = convert(Integer, (size(m, 1) - 1)/p)

    !(p == size(m, 2) == size(S, 1)) && throw(DimensionMismatch("y is a $T x $p, m is a $(size(m, 1)) x $(size(m, 2)) and S is a $(size(S, 1)) x $(size(S, 2))."))

    estim = Estimation(y, d, priors.β, priors.δ)

    F = ones(d*p + 1)
    for t = 1+d:T
        if d > 0
            F[2:end] = vec(priors.y[t-d:t-1, :])
        end
        # Predict at time t
        R = P / priors.δ
        Q = F' * R * F + 1.0
        μ = m' * F
        Σ = Q * (1 - priors.β) / (3priors.β*priors.k - 2priors.k) * S
        e = y[t, :] - μ
        S = S / priors.k

        estim.m[t, :, :] = m
        estim.P[t, :, :] = R
        estim.S[t, :, :] = S
        estim.Ω[t, :, :] = kron(S, R)
        
        estim.μ[t, :]    = μ
        estim.Σ[t, :, :] = Σ
        estim.e[t, :]    = e
        estim.u[t, :]    = inv(cholesky(Σ).L) * e

        # Update at time t
        K = R * F / Q
        m = m + K * e'
        P = R - K * K' * Q
        S = S + e*e'/Q
    end

    # Predict at time T + 1
    if d > 0
        F[2:end] = vec(priors.y[T-d+1:T, :])
    end
    # Predict at time t
    R = P / priors.δ
    Q = F' * R * F + 1.0
    μ = m' * F
    Σ = Q * (1 - priors.β) / (3priors.β*priors.k - 2priors.k) * S

    estim.m[T + 1, :, :] = m
    estim.P[T + 1, :, :] = R
    estim.S[T + 1, :, :] = S
    estim.Ω[T + 1, :, :] = kron(S, R)
        
    estim.μ[T + 1, :]    = μ
    estim.Σ[T + 1, :, :] = Σ

    return estim
end