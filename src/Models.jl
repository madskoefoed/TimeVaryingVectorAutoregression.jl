mutable struct StochasticVolatilityModel
    priors::Vector{Priors}
    posteriors::Vector{Posteriors}
    priorpredictive::Vector{PriorPredictive}
    y::FLOATMAT
    e::FLOATMAT
    z::FLOATMAT
    ll::FLOATVEC
    const hyperparameters::Hyperparameters
    const p::Integer

    function StochasticVolatilityModel(priors::Priors, hyperparameters::Hyperparameters)
        y = FLOATMAT[]
        e = FLOATMAT[]
        z = FLOATMAT[]
        ll = FLOATVEC[]

        priors = Priors[]
        posteriors = Posteriors[]

        priorpredictive = PriorPredictive[]
        #posteriorpredictive = MvTDist[]

        p = size(priors.S, 1)
        
        new(priors, posteriors, priorpredictive, y, e, z, ll, hyperparameters, p, k)
    end
end