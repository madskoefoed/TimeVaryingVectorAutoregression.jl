
mutable struct Priors
    y::Array{<:Real, 2} # T x p
    m::Array{<:Real, 2} # dp + 1 x p
    P::Array{<:Real, 2} # dp + 1 x dp + 1
    S::Array{<:Real, 2} # p x p
    β::Real
    δ::Real
    function Priors(y, m, P, S, β, δ)
        str = "y is $(size(y, 1)) x $(size(y, 2)),\nm is $(size(m, 1)) x $(size(m, 2)),\nP is $(size(P, 1)) x $(size(P, 2)), \nS is $(size(S, 1)) x $(size(S, 2))."

        !(size(m, 1) == size(P, 1) == size(P, 2)) && throw(DimensionMismatch(str))
        !(size(y, 2) == size(m, 2) == size(S, 1) == size(S, 2)) && throw(DimensionMismatch(str))

        any(diag(S) .<= 0) && throw(ArgumentError("The diagonal elements of S must be strictly positive."))
        any(diag(P) .<= 0) && throw(ArgumentError("The diagonal elements of P must be strictly positive."))

        (δ  > 0 && δ <= 1) || throw(ArgumentError("0 < δ ≤ 1 required (currently $δ)."))
        (β > 2/3 && β < 1) || throw(ArgumentError("$(2//3) < β < 1 required (currently $β)."))

        return new(y, m, P, S, β, δ)
    end
end

Prior(y::Vector{T}, m::Matrix{T}, P::Matrix{T}, S::Matrix{T}, β::T, δ::T) where T<:Real = Prior(vec2mat(y), m, P, S, β, δ)
Prior(y::Vector{T}, m::Vector{T}, P::Matrix{T}, S::T, β::T, δ::T) where T<:Real = Prior(vec2mat(y), vec2mat(m), P, num2mat(S), β, δ)
Prior(y::Vector{T}, m::T, P::Matrix{T}, S::T, β::T, δ::T) where T<:Real = Prior(vec2mat(y), num2mat(m), P, num2mat(S), β, δ)

mutable struct Estimation
    y::Array{<:Real, 2} # T x p
    m::Array{<:Real, 3} # T x dp + 1 x p
    P::Array{<:Real, 3} # T x dp + 1 x dp + 1
    S::Array{<:Real, 3} # T x p x p
    β::Real
    δ::Real
    μ::Array{<:Real, 2} # T x p
    Σ::Array{<:Real, 3} # T x p x p
    #h::Integer
end

#mutable struct Forecast
#    μ::Array{<:Real, 2} # T x p
#    Σ::Array{<:Real, 3} # T x p x p
#    h::Integer
#end