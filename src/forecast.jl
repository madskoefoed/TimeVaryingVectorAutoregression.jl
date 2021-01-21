
"""
forecast(Estimation)

TODO
"""

function forecast(estim::Estimation)#, h=1::Integer)
    #h > 0 || throw(ArgumentError("The horizon parameter (h = $h) must be a strictly positive integer."))
    T, p = size(estim.y)
    m = estim.m
    P = estim.P
    S = estim.S
    d = convert(Integer, (size(m, 2) - 1)/p)
    ν = estim.β/(1 - estim.β)
    k = (estim.β - p*estim.β + p)/(2*estim.β - p*estim.β + p - 1)
    
    #μ = zeros(eltype(estim.m), T + 1, p)
    #Σ = zeros(eltype(estim.m), T + 1, p, p)

    μ = zeros(T + 1, p)
    Σ = zeros(T + 1, p, p)

    F = ones(d*p + 1)
    for t = 1+d:T+1
        if d > 0
            F[2:end] = vec(priors.y[t-d:t-1, :]) 
        end
        μ[t, :] = predicted_mean(F, m)
        Σ[t, :, :] = predicted_cov(F, P, S, β, δ, k)
    end
    return (μ = μ, Σ = Σ, ν = ν)
end