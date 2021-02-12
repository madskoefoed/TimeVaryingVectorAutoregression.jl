
num2mat(x::Real) = fill(x, 1, 1)
vec2mat(x::Vector) = repeat(x, 1, 1)

get_ν(β) = β/(1 - β)
get_β(ν) = ν/(1 + ν)
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

#function predict_step!(y, F, priors)
#    R = P.priors / priors.δ
#    Q = F' * R * F + 1.0
#    S = S / priors.k
#    μ = priors.m' * F
#    Σ = Q * (1 - priors.β) / (3priors.β - 2) * priors.S
#    e = y[t, :] - μ
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
