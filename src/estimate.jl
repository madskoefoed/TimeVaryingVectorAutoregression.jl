function estimate!(model::StochasticVolatilityModel, y::FLOATMAT)
    for t in axes(y, 1)
        estimate!(model, y[t, :])
    end

    return nothing
end

function estimate!(model::StochasticVolatilityModel, y::FLOATVEC)
    # Predict at time t|t-1
    priors = Priors(model.priors[end], model.hyperparameters)

    ppd = PriorPredictive(model.priors[end], model.hyperparameters)

    push!(model.priors, priors)

    # Prediction error
    e = error(y, ppd)
    z = standardised_error(e, ppd)

    push!(model.e, e)
    push!(model.z, z)

    # Predictive log-likelihood
    push!(model.loglik, logpdf(ppd, y))

    # Update at time t|t
    post = Posteriors(priors, e)
    ppd  = PosteriorPredictive(post, model.hyperparameters)
    
    
    push!(model.posteriors, posteriors)
    #model.posteriorpredictive = push!(model.posteriorpredictive, ppd)


    return nothing
end