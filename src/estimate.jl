
"""
estimate(Priors)

Estimate a stochastic volatility model. Volatility is modelled as a random walk.

The hyperparameters 2/3 < β < 1 and 0 < δ ≤ 1 are discount factors, controlling the shocks to the Σ and Ω respectively.
"""

function estimate(priors::Priors)
    m = priors.m
    P = priors.P
    S = priors.S

    T, p = size(priors.y)
    d = convert(Integer, (size(m, 1) - 1)/p)
    #n = 1/(1 - priors.β)
    #ν = priors.β*n
    k = (priors.β - p*priors.β + p)/(2priors.β - p*priors.β + p - 1)

    estim = Estimation(priors.y,
                       repeat(reshape(m, 1, size(m, 1), size(m, 2)), T + 1, 1, 1),
                       repeat(reshape(P, 1, size(P, 1), size(P, 2)), T + 1, 1, 1),
                       repeat(reshape(S, 1, size(S, 1), size(S, 2)), T + 1, 1, 1),
                       priors.β,
                       priors.δ)

    F = ones(d*p + 1)
    for t = 1+d:T
        if d > 0
            F[2:end] = vec(priors.y[t-d:t-1, :])
        end
        m, P, S = update_priors(y[t, :], F, m, P, S, priors.δ, k)

        estim.m[t + 1, :, :] = m
        estim.P[t + 1, :, :] = P
        estim.S[t + 1, :, :] = S
    end
    return estim
end