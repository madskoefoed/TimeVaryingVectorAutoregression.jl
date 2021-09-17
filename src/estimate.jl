function estimate(model::Model)
    y, m, P, S, β, δ = model.y, model.m, model.P, model.S, model.β, model.δ

    T, p = size(y)
    d = convert(Integer, (size(m, 1) - 1)/p)
    a = d*p + 1
    F = [1; zeros(a - 1)]
    k = get_k(p, β)
    ν = get_ν(β)
    
    out = Output(y,               # y
                 zeros(T, a, p),  # m
                 zeros(T, a, a),  # P
                 zeros(T, p, p),  # S
                 zeros(T, p),     # μ
                 zeros(T, p, p),  # Σ
                 zeros(T, p),     # e
                 zeros(T, p),     # u
                 ν,
                 β,
                 δ)

    for t = 1:T
        if d > 1 && t > d
            F[2:end] = vec(y[t-1:-1:t-d, :])
        end
        
        # Predict parameters at time t|t-1
        P, S, Q = predict_parameters(P, S, F, δ, k)
        
        # Predict at time t|t-1
        μ, Σ = predict(m, Q, S, F, β)

        # Prediction error
        e = error(y[t, :], μ)
        u = standardised_error(e, Σ)

        # Storage
        out.m[t, :, :] = m
        out.P[t, :, :] = P
        out.S[t, :, :] = S
        out.μ[t, :]    = μ
        out.Σ[t, :, :] = Σ
        out.e[t, :]    = e
        out.u[t, :]    = u

        # Update at time t|t
        if t > d
            m, P, S = update_parameters(m, P, Q, S, F, e)
        end

    end
    return out
end

function predict_parameters(P, S, F, δ, k)
    # note: m is not updated
    P = P / δ
    Q = F' * P * F + 1.0
    S = S / k
    return (P, S, Q)
end

function predict(m, Q, S, F, β)
    μ = m' * F
    Σ = Q * (1 - β) / (3β - 2) * S
    return (μ, Σ)
end

function error(y, μ)
    return y - μ
end

function standardised_error(e, Σ)
    return inv(cholesky(Σ).L) * e
end

function update_parameters(m, P, Q, S, F, e)
    K = P * F / Q
    m = m + K * e'
    P = P - K * K' * Q
    S = S + e*e'/Q
    return (m, P, S)
end