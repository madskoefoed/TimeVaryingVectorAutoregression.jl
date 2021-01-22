
#mutable struct Hyperparameters
#    β::Real
#    δ::Real
#    function Hyperparameters(β, δ)
#        (δ  > 0 && δ <= 1) || throw(ArgumentError("0 < δ ≤ 1 required (currently $δ)."))
#        (β > 2/3 && β < 1) || throw(ArgumentError("$(2//3) < β < 1 required (currently $β)."))
#    return new(β, δ)
#end

mutable struct Priors
    m::Array{<:Real, 2} # dp + 1 x p
    P::Array{<:Real, 2} # dp + 1 x dp + 1
    S::Array{<:Real, 2} # p x p
    β::Real
    δ::Real
    n::Real
    ν::Real
    k::Real
    
    function Priors(m, P, S, β, δ)
        p = size(S, 1)
        str = "m is $(size(m, 1)) x $(size(m, 2)),\nP is $(size(P, 1)) x $(size(P, 2)), \nS is $(size(S, 1)) x $(size(S, 2))."

        !(size(m, 1) == size(P, 1) == size(P, 2)) && throw(DimensionMismatch(str))
        !(size(m, 2) == size(S, 1) == size(S, 2)) && throw(DimensionMismatch(str))

        any(diag(S) .<= 0) && throw(ArgumentError("The diagonal elements of S must be strictly positive."))
        any(diag(P) .<= 0) && throw(ArgumentError("The diagonal elements of P must be strictly positive."))

        (δ  > 0 && δ <= 1) || throw(ArgumentError("0 < δ ≤ 1 required (currently $δ)."))
        (β > 2/3 && β < 1) || throw(ArgumentError("$(2//3) < β < 1 required (currently $β)."))

        n = 1/(1 - β)
        ν = get_ν(β)
        k = get_k(p, β)
        return new(m, P, S, β, δ, n, ν, k)
    end
end

#Prior(y::Vector{T}, m::Matrix{T}, P::Matrix{T}, S::Matrix{T}, β::T, δ::T) where T<:Real = Prior(vec2mat(y), m, P, S, β, δ)
#Prior(y::Vector{T}, m::Vector{T}, P::Matrix{T}, S::T, β::T, δ::T) where T<:Real = Prior(vec2mat(y), vec2mat(m), P, num2mat(S), β, δ)
#Prior(y::Vector{T}, m::T, P::Matrix{T}, S::T, β::T, δ::T) where T<:Real = Prior(vec2mat(y), num2mat(m), P, num2mat(S), β, δ)

mutable struct Estimation
    y::Array{<:Real, 2} # T x p
    m::Array{<:Real, 3} # T x dp + 1 x p
    P::Array{<:Real, 3} # T x dp + 1 x dp + 1
    S::Array{<:Real, 3} # T x p x p
    Ω::Array{<:Real, 3}
    β::Real
    δ::Real
    μ::Array{<:Real, 2} # T x p
    Σ::Array{<:Real, 3} # T x p x p
    ν::Real
    e::Array{<:Real}
    u::Array{<:Real}
end

function Estimation(y::Matrix{<:Real}, d::Integer, β::Real, δ::Real)
    T, p = size(y)
    a = d*p + 1
    return Estimation(zeros(T, p),          # y
                      zeros(T, a, p),       # m
                      zeros(T, a, a),       # P
                      zeros(T, p, p),       # S
                      zeros(T, a*p, a*p),   # Ω
                      β,
                      δ,
                      zeros(T, p),          # μ
                      zeros(T, p, p),       # Σ
                      get_ν(β),             # ν - degrees of freedom
                      zeros(T, p),          # e - prediction error
                      zeros(T, p))          # u - scaled prediction error
end