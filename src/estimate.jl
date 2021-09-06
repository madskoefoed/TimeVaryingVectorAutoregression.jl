function estimate(model::Model)
    y, m, P, S, β, δ = model.y, model.m, model.P, model.S, model.β, model.δ

    T, p = size(y)
    d = convert(Integer, (size(m, 1) - 1)/p)
    a = d*p + 1
    F = ones(a)
    k = get_k(p, β)
    ν = get_ν(β)

    out = Output(y[(d + 1):end, :],
                 zeros(T-d, a, p),
                 zeros(T-d, a, a),
                 zeros(T-d, p, p),
                 zeros(T-d, p),
                 zeros(T-d, p, p),
                 zeros(T-d, p),
                 zeros(T-d, p),
                 ν,
                 δ)

    for t = 1:T
        if t > d
            F[2:end] = vec(y[t-d:t-1, :])
        end
        # Predict at time t
        P = P / δ
        Q = F' * P * F + 1.0
        S = S / k
        μ = m' * F
        Σ = Q * (1 - β) / (3β - 2) * S
        e = y[t, :] - μ

        out.m[t, :, :] = m
        out.P[t, :, :] = P
        out.S[t, :, :] = S
        out.μ[t, :]    = μ
        out.Σ[t, :, :] = Σ
        out.e[t, :]    = e
        out.u[t, :]    = inv(cholesky(Σ).L) * e

        # Update at time t
        K = P * F / Q
        m = m + K * e'
        P = P - K * K' * Q
        S = S + e*e'/Q
    end
    return out
end