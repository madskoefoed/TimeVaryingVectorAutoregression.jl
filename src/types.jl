mutable struct Model
    y::FLOATMAT # T x p
    m::FLOATMAT # dp + 1 x p
    P::FLOATMAT # dp + 1 x dp + 1
    S::FLOATMAT # p x p
    β::Real
    δ::Real
    function Model(y, m, P, S, β, δ)
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

mutable struct Output
    y::FLOATMAT # T x p
    m::FLOATARR # T x dp + 1 x p
    P::FLOATARR # T x dp + 1 x dp + 1
    S::FLOATARR # T x p x p
    μ::FLOATMAT # T x p
    Σ::FLOATARR # T x p x p
    e::FLOATMAT # T x p
    u::FLOATMAT # T x p
    ν::Real
    β::Real
    δ::Real
end

mutable struct Simulation
    y::FLOATMAT # T x p
    Ω::FLOATARR # T x dp + 1 x p
    m::FLOATMAT # dp + 1 x p
    P::FLOATMAT # dp + 1 x dp + 1
    S::FLOATMAT # p x p
    T::Integer
    function Simulation(m, P, S, T)
        str = "y is $T x $(size(m, 2)),\nm is $(size(m, 1)) x $(size(m, 2)),\nP is $(size(P, 1)) x $(size(P, 2)), \nS is $(size(S, 1)) x $(size(S, 2))."

        y = zeros(T, size(m, 2))
        Ω = zeros(T, size(m, 1), size(m, 2))

        !(size(m, 1) == size(P, 1) == size(P, 2)) && throw(DimensionMismatch(str))
        !(size(m, 2) == size(S, 1) == size(S, 2)) && throw(DimensionMismatch(str))

        any(diag(S) .<= 0) && throw(ArgumentError("The diagonal elements of S must be strictly positive."))
        any(diag(P) .<= 0) && throw(ArgumentError("The diagonal elements of P must be strictly positive."))

        return new(y, Ω, m, P, S, T)
    end
end