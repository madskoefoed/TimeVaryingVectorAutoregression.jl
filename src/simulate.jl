function simulate(T::Integer, Φ::FLOATVEC, Σ::PDMat)
    @assert T > 0 "The number of observations must be strictly positive."
    
    p = length(Φ)
    str = "Φ is a $(p)-dimensional vector,\nwhile Σ is a $(size(Σ, 1)) x $(size(Σ, 2)) matrix."
    !(p == size(Σ, 1) == size(Σ, 2)) && throw(DimensionMismatch(str))

    y = rand(MvNormal(Φ, Σ), T)

    return y'
end