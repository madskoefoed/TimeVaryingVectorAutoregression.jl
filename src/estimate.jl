
"""
estimate(model::MSV)

Estimate a stochastic volatility model. Volatility is modelled as a random walk.

The hyperparameter 2/3 < β < 1 is a discount factor controlling the shocks to Σ.
"""

function estimate(model::MSV)
    y, S, β, ν = model.y, model.S, model.β, model.ν

    T, p = size(y)
    k = get_k(p, β)

    estim = Estimation(model)
    for t = 1:T
        # Predict at time t
        S = S / k
        Σ = (1 - β) / (3β - 2) * S

        estim.S[t, :, :] = S
        estim.Σ[t, :, :] = Σ
        estim.e[t, :]    = y[t, :]
        estim.u[t, :]    = inv(cholesky(Σ).L) * y[t, :]

        # Update at time t
        S = S + y[t, :]*y[t, :]'
    end
    return estim
end

"""
estimate(model::TVVAR)

Estimate a stochastic volatility model. Volatility is modelled as a random walk.

The hyperparameters 2/3 < β < 1 and 0 < δ ≤ 1 are discount factors, controlling the shocks to the Σ and Ω respectively.
"""


function estimate(model::TVVAR)
    y, m, P, S, β, δ, ν = model.y, model.m, model.P, model.S, model.β, model.δ, model.ν

    T, p = size(y)
    d = convert(Integer, (size(m, 1) - 1)/p)
    F = ones(d*p + 1)
    k = get_k(p, β)

    estim = Estimation(model)
    if d > 0
        
    end

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