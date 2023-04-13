function estimate_batch!(model::TVVAR)
    T, p, d, k = model.T, model.p, model.d, model.k
    m, P, S = model.priors.m, model.priors.P, model.priors.S
    β, δ = model.hyperparam.β, model.hyperparam.δ
    y = model.y

    F = ones(1+d*p)
    for t in (d + 1):T
        if d > 0
            F[2:end] = vec(y[t-d:t-1, :])
        end

        # Predict parameters at time t|t-1
        P, S, Q = predict_parameters(P, S, F, δ, k)

        model.Q[t] = Q

        # Predict at time t|t-1
        μ, Σ = predict_outcome(m, Q, S, F, β)

        # Prediction error
        e = model.y[t, :] - μ
        u = standardised_error(e, Σ)

        # Predictive log-likelihood
        ll = logpdf(MvTDist(model.hyperparam.ν, μ, PDMat(Σ)), model.y[t, :])

        model.loglik[t] = model.loglik[t-1] + ll

        # Update at time t|t
        m, P, S = update_parameters(m, P, Q, S, F, e)

        # Storage
        model.m[t, :, :], model.P[t, :, :], model.S[t, :, :] = m, P, S
        model.μ[t, :], model.Σ[t, :, :], model.e[t, :], model.u[t, :] = μ, Σ, e, u
    end
end

function predict!(model::TVVAR)
    T, p, d, k = model.T, model.p, model.d, model.k
    m, P, S = model.m[T, :, :], model.P[T, :, :], model.S[T, :, :]
    β, δ = model.hyperparam.β, model.hyperparam.δ
    y = model.y

    F = ones(1+d*p)
    if d > 0
        F[2:end] = vec(y[T-d+1:T, :])
    end

    # Predict parameters at time t|t-1
    P, S, Q = predict_parameters(P, S, F, δ, k)
    
    push!(model.Q, Q)

    # Predict at time t|t-1
    μ, Σ = predict_outcome(m, Q, S, F, β)

    # Expand containers
    model.m = vcat(model.m, reshape(m, (1, d*p+1, p)))
    model.P = vcat(model.P, reshape(P, (1, d*p+1, d*p+1)))
    model.S = vcat(model.S, reshape(S, (1, p, p)))

    model.μ = vcat(model.μ, μ')
    model.Σ = vcat(model.Σ, reshape(Σ, (1, p, p)))
end

function update!(model::TVVAR, y::FLOATVEC)
    # Update T
    model.T += 1
    
    T, p, d = model.T, model.p, model.d

    F = ones(1+d*p)
    if d > 0
        F[2:end] = vec(model.y[T-d:T-1, :])
    end

    # Expand y
    model.y = vcat(model.y, y')

    # Predictions at time t|t-1
    μ = model.μ[T, :]
    Σ = PDMat(model.Σ[T, :, :])

    # Prediction error
    e = y - μ
    u = standardised_error(e, Σ)

    ll = logpdf(MvTDist(model.hyperparam.ν, μ, Σ), y)
    ll = model.loglik[T-1] + ll

    # Update at time t|t
    model.m[T, :, :], model.P[T, :, :], model.S[T, :, :] = update_parameters(model.m[T, :, :],
                                                                             model.P[T, :, :],
                                                                             model.Q[T],
                                                                             model.S[T, :, :],
                                                                             F,
                                                                             e)

    push!(model.loglik, ll)

    model.e = vcat(model.e, e')
    model.u = vcat(model.u, u')
end

function predict_parameters(P, S, F, δ, k)
    P = P / δ
    Q = F' * P * F + 1.0
    S = S / k
    return (P, S, Q)
end

function update_parameters(m, P, Q, S, F, e)
    K = P * F / Q
    m = m + K * e'
    P = P - K * K' * Q
    S = S + e*e'/Q
    return (m, P, S)
end

function predict_outcome(m, Q, S, F, β)
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