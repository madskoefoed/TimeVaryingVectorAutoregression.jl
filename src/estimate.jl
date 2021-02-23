
"""
estimate(Priors)

Estimate a stochastic volatility model. Volatility is modelled as a random walk.

The hyperparameters 2/3 < β < 1 and 0 < δ ≤ 1 are discount factors, controlling the shocks to the Σ and Ω respectively.
"""

function estimate(model::TVVAR)
    y, m, P, S, β, δ, ν = model.y, model.m, model.P, model.S, model.β, model.δ, model.ν

    T, p = size(y)
    d = convert(Integer, (size(m, 1) - 1)/p)
    F = ones(d*p + 1)
    k = get_k(p, β)

    str = "y is $T x $p,\nm is $(size(m, 1)) x $(size(m, 2)),\nP is $(size(P, 1)) x $(size(P, 2)), \nS is $(size(S, 1)) x $(size(S, 2))."

    !(size(m, 1) == size(P, 1) == size(P, 2)) && throw(DimensionMismatch(str))
    !(p == size(m, 2) == size(S, 1) == size(S, 2)) && throw(DimensionMismatch(str))

    estim = Estimation(model)
    for t = 1+d:T
        if d > 0
            F[2:end] = vec(y[t-d:t-1, :])
        end
        # Predict at time t
        P = P / δ
        Q = F' * P * F + 1.0
        S = S / k
        μ = m' * F
        Σ = Q * (1 - β) / (3β - 2) * S
        e = y[t, :] - μ

        estim.m[t, :, :] = m
        estim.P[t, :, :] = P
        estim.S[t, :, :] = S
        estim.Ω[t, :, :] = kron(S, P)
        
        estim.μ[t, :]    = μ
        estim.Σ[t, :, :] = Σ
        estim.e[t, :]    = e
        estim.u[t, :]    = inv(cholesky(Σ).L) * e

        # Update at time t
        K = P * F / Q
        m = m + K * e'
        P = P - K * K' * Q
        S = S + e*e'/Q
    end
    return estim
end