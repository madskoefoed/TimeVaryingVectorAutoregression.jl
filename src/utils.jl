
num2mat(x::Real) = fill(x, 1, 1)
vec2mat(x::Vector) = repeat(x, 1, 1)

function diagnostics(y, μ, Σ, e, u)
    ME = mean(e; dims = 1)
    MAE = mean(abs.(e); dims = 1)
    MSSE = mean(u.^2; dims = 1)
    LL = mean([logpdf(MvTDist(ssm.ν, μ[t, :], Σ[t, :, :]), y[t, :]) for t in 1:size(ssm.y, 1)])
    return (ME = ME, MAE = MAE, MSSE = MSSE, LogLik = LL)
end

# Predict at time t-1
#Σ = Q * (1 - β) / (3β*k - 2k) * S
#Σ = Q * (1 - β) / (2β - 1) * S
#u = inv(cholesky(Σ).L) * e

function update_priors(y, F, m, P, S, δ, k)
    R = P / δ
    Q = F' * R * F + 1.0
    μ = m' * F
    Σ = Q * (1 - β) / (3β*k - 2k) * S
    e = y - μ
    K = R * F / Q
    m = m + K * e'
    P = R - K * K' * Q
    S = S/k + e*e'/Q
    return (m, P, S, μ, Σ)
end

#predicted_mean(F, m) = m' * F
#predicted_cov(F, P, S, β, δ, k) = (F' * (P / δ) * F + 1.0) * (1 - β) / (3β*k - 2k) * S