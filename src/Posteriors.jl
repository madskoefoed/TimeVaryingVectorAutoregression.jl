mutable struct Posteriors
    m::FLOATVEC
    P::AbstractFloat
    S::PDMat

    function Posteriors(priors::Priors, error::FLOATVEC)
        m, P, S = posterior_param(priors, error)

        new(m, P, S)
    end
end

mutable struct PosteriorPredictive
    μ::FLOATVEC
    Σ::PDMat

    function PosteriorPredictive(posteriors::Posteriors, hyper::Hyperparameters)
        μ, Σ = posterior_pred(posteriors, hyper)

        new(μ, Σ)
    end
end

function posterior_param(priors::Priors, error::FLOATVEC)
    Q = priors.P + 1
    K = priors.P / Q
    m = priors.m + K * error'
    P = priors.P - K * K' * Q
    S = priors.S + error*error'/Q
    
    return (m, P, S)
end

function posterior_pred(post::PosteriorPredictive, hyper::Hyperparameters)
    μ = post.m
    Σ = PDMat((post.P + 1) * (1 - hyper.β) / (2*hyper.β - 1) * post.S)

    return (μ, Σ)
end

prior_param(post::Posteriors, hyper::Hyperparameters) = prior_param(post.m, post.P, post.S, hyper::Hyperparameters)