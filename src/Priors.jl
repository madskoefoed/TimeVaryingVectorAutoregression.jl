mutable struct Priors
    m::FLOATVEC
    P::AbstractFloat
    S::PDMat

    function Priors(m::FLOATVEC, P::AbstractFloat, S::PDMat, hyper::Hyperparameters)
        str = "m is a $(length(m))-dimensional vector, but S is a $(size(S, 1)) x $(size(S, 2)) matrix."

        !(length(m) == size(S, 1) == size(S, 2)) && throw(DimensionMismatch(str))
    
        any(diag(S) .<= 0) && throw(ArgumentError("The diagonal elements of S must be strictly positive."))
        any(P <= 0) && throw(ArgumentError("P must be strictly positive."))

        m, P, S = prior_param(m, P, S, hyper)

        new(m, P, S)
    end
end

mutable struct PriorPredictive
    μ::FLOATVEC
    Σ::PDMat

    function PriorPredictive(priors::Priors, hyper::Hyperparameters)
        μ, Σ = prior_pred(priors, hyper)

        new(μ, Σ)
    end
end

function prior_param(m::FLOATVEC, P::AbstractFloat, S::PDMat, hyper::Hyperparameters)
    p = size(S, 1)
    k = (hyper.β - p*hyper.β + p)/(hyper.β - p*hyper.β + p - 1)

    P = P / hyper.δ
    S = S / k

    return (m, P, S)
end

function prior_pred(priors::Priors, hyper::Hyperparameters)
    μ = priors.m
    Σ = PDMat((priors.P + 1) * (1 - hyper.β) / (3*hyper.β - 2) * priors.S)

    return (μ, Σ)
end

error(y::FLOATVEC, μ::FLOATVEC) = y - μ
error(y::FLOATVEC, prior_pred::PriorPredictive) = y - prior_pred.μ

standardised_error(e, Σ) = inv(cholesky(Σ).L) * e
standardised_error(e, prior_pred::PriorPredictive) = inv(cholesky(prior_pred.Σ).L) * e