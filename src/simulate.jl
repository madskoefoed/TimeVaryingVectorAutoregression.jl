
"""
simulate(StateSpaceModel)

Simulate a stochastic volatility model. Volatility is NOT currently implemented as a random walk, but rather given by Σ.
"""

function simulate!(model::TVVAR)
    y, m, P, S, β, δ = model.y, model.m, model.P, model.S, model.β, model.δ

    T, p = size(y)
    if size(m, 1) == 1
        a = 0
    else
        a = convert(Int, size(m, 1) - 1 / p)
    end
    d = 1 + a * p
    n = 1/(1 - β)

    # Priors (t = 0)
    Σ = rand(InverseWishart(n + 2p, S))
    Φ = rand(MatrixNormal(m, P, Σ))
    F = ones(AbstractFloat, d)
    O = zeros(T + 1, d, p)
    O[1, :, :] = Φ
    for t = 1:T
        if t > a & a > 0
            F[2:end] = vec(y[(t - a):(t - 1), :])
        end

        # Sigma
        Σ = rand(InverseWishart(n + 2p, S))

        # State equation
        Φ = Φ + rand(MatrixNormal(zeros(d, p), P, Σ))
        O[t + 1, :, :] = Φ

        # Measurement (observation) equation
        y[t, :] = F' * Φ + rand(MvNormal(Σ))'
    end
    model.y = y
    return O
end

function simulate!(model::VAR)
    y, m, Σ = model.y, model.m, model.Σ

    T, p = size(y)
    if size(m, 1) == 1
        a = 0
    else
        a = convert(Int, size(m, 1) - 1 / p)
    end
    d = 1 + a * p
    n = 1/(1 - β)

    # Priors (t = 0)
    Σ = rand(InverseWishart(n + 2p, S))
    Φ = rand(MatrixNormal(m, P, Σ))
    F = ones(AbstractFloat, d)
    O = zeros(T + 1, d, p)
    O[1, :, :] = Φ
    for t = 1:T
        if t > a & a > 0
            F[2:end] = vec(y[(t - a):(t - 1), :])
        end

        # Sigma
        Σ = rand(InverseWishart(n + 2p, S))

        # State equation
        Φ = Φ + rand(MatrixNormal(zeros(d, p), P, Σ))
        O[t + 1, :, :] = Φ

        # Measurement (observation) equation
        y[t, :] = F' * Φ + rand(MvNormal(Σ))'
    end
    model.y = y
    return O
end