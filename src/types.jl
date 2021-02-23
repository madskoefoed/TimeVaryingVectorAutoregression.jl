mutable struct TVVAR
    y::Array{<:Real, 2} # T x p
    m::Array{<:Real, 2} # dp + 1 x p
    P::Array{<:Real, 2} # dp + 1 x dp + 1
    S::Array{<:Real, 2} # p x p
    β::Real
    δ::Real
    ν::Real

    function TVVAR(y, m, P, S, β, δ)
        T, p = size(y)
        str = "y is $(size(y, 1)) x $(size(y, 2)),\nm is $(size(m, 1)) x $(size(m, 2)),\nP is $(size(P, 1)) x $(size(P, 2)), \nS is $(size(S, 1)) x $(size(S, 2))."

        !(size(m, 1) == size(P, 1) == size(P, 2)) && throw(DimensionMismatch(str))
        !(size(y, 2) == size(m, 2) == size(S, 1) == size(S, 2)) && throw(DimensionMismatch(str))

        any(diag(S) .<= 0) && throw(ArgumentError("The diagonal elements of S must be strictly positive."))
        any(diag(P) .<= 0) && throw(ArgumentError("The diagonal elements of P must be strictly positive."))

        (δ  > 0 && δ <= 1) || throw(ArgumentError("0 < δ ≤ 1 required (currently $δ)."))
        (β > 2/3 && β < 1) || throw(ArgumentError("$(2//3) < β < 1 required (currently $β)."))

        ν = get_ν(β)
        return new(y, m, P, S, β, δ, ν)
    end
end

mutable struct Estimation
    y::Array{<:Real, 2} # T x p
    m::Array{<:Real, 3} # T x dp + 1 x p
    P::Array{<:Real, 3} # T x dp + 1 x dp + 1
    S::Array{<:Real, 3} # T x p x p
    Ω::Array{<:Real, 3} # T x p(dp + 1) x p(dp + 1)
    μ::Array{<:Real, 2} # T x p
    Σ::Array{<:Real, 3} # T x p x p
    ν::Real
    δ::Real
    e::Array{<:Real, 2} # T x p
    u::Array{<:Real, 2} # T x p
end

function Estimation(priors::TVVAR)
    T, p = size(priors.y)
    a = size(priors.m, 1)
    return Estimation(priors.y,           # y
                      zeros(T, a, p),     # m
                      zeros(T, a, a),     # P
                      zeros(T, p, p),     # S
                      zeros(T, a*p, a*p), # Ω
                      zeros(T, p),        # μ
                      zeros(T, p, p),     # Σ
                      priors.ν,           # ν
                      priors.δ,           # δ
                      zeros(T, p),        # e
                      zeros(T, p))        # u
end