
num2mat(x::Real) = fill(x, 1, 1)
vec2mat(x::Vector) = repeat(x, 1, 1)

get_ν(β) = β/(1 - β)
get_k(p, β) = (β - p*β + p)/(2β - p*β + p - 1)

#prior_mean(F, m) = m' * F
#prior_covariance(Q, S, β) = Q * (1 - β) / (3β*get_k(size(S, 1), β) - 2*get_k(size(S, 1), β)) * S
#posterior_covariance(Q, S, β) = Q * (1 - β) / (2β - 1) * S

function goodness_of_fit(estim::Estimation)
    ME = mean(estim.e; dims = 1)
    MAE = mean(abs.(estim.e); dims = 1)
    RMSE = sqrt.(mean(estim.e.^2; dims = 1))
    MSSE = mean(estim.u.^2; dims = 1)
    LL = mean([logpdf(MvTDist(estim.ν, estim.μ[t, :], estim.Σ[t, :, :]), estim.y[t, :]) for t in 1:size(estim.y, 1)])
    return (ME = ME, MAE = MAE, RMSE = RMSE, MSSE = MSSE, LogLik = LL)
end

# Predict at time t-1
#Σ = Q * (1 - β) / (3β*k - 2k) * S
#Σ = Q * (1 - β) / (2β - 1) * S
#u = inv(cholesky(Σ).L) * e

#function predict_step(y, F, m, P, S, β, δ, k)
#    R = P / δ
#    Q = F' * R * F + 1.0
#    μ = m' * F
#    Σ = Q * (1 - β) / (3β*k - 2k) * S
#    e = y - μ
#    S = S/k
#    return (R, Q, S, μ, Σ, e)
#end

#function update_step(e, F, R, Q, m, R, Q, S)
#    K = R * F / Q
#    m = m + K * e'
#    P = R - K * K' * Q
#    S = S + e*e'/Q
#    return (m, P, S)
#end

#function update_priors(y, F, m, P, S, β, δ, k)
#    R = P / δ
#    Q = F' * R * F + 1.0
#    μ = m' * F
#    Σ = Q * (1 - β) / (3β*k - 2k) * S
#    e = y - μ
#    K = R * F / Q
#    m = m + K * e'
#    P = R - K * K' * Q
#    S = S/k + e*e'/Q
#    return (m, P, S, μ, Σ)
#end

# Local level multivariate Kalman filter
function kalman_filter(y::Matrix{<:Real}, R = 1.0, Q = 1.0)
    T, p = size(y)

    m = zeros(p)
    P = Matrix(1000.0*I, p, p)
    R = Matrix(1.0*I, p, p) .* R
    Q = Matrix(0.01*I, p, p) .* Q

    out = (m = zeros(T + 1, p), P = fill(1000.0, T + 1, p, p), S = fill(1.0, T + 1, p, p), e = zeros(T, p))

    for t = 1:T
        # Predict
        m = m
        P = P + Q

        # Update
        S = P + R
        K = P * inv(R + P)
        e = y[t, :] - m
        m = m + K * e
        P = P - K * P

        out.e[t, :] = e

        out.m[t + 1, :]    = m
        out.P[t + 1, :, :] = P
        out.S[t + 1, :, :] = S

    end

    ll = sum([logpdf(Normal(out.m[t, 1], out.S[t, 1, 1]), y[t, 1]) for t = 1:T])
    return out, ll
end

kf = kalman_filter(y)
ss = statespace(local_level(y))

opt = [kalman_filter(y, 2.0^i, 2.0^j)[2] for i in -5:10, j in -5:10]

kf = kalman_filter(y, 2.0^7, 2.0^4)

scatter(y[:, 1], label = "Observations")
plot!(kf[1].m[:, 1], label = "KF")
plot!(ss.filter.a[:, 1, 1], label = "StateSpace")