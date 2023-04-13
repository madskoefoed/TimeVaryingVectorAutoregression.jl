abstract type TimeVaryingModels end

mutable struct Priors
    m::FLOATMAT
    P::FLOATMAT
    S::FLOATMAT
    function Priors(m::FLOATMAT, P::FLOATMAT, S::FLOATMAT)

        str = "m is $(size(m, 1)) x $(size(m, 2)),\nP is $(size(P, 1)) x $(size(P, 2)), \nS is $(size(S, 1)) x $(size(S, 2))."

        !(size(m, 1) == size(P, 1) == size(P, 2)) && throw(DimensionMismatch(str))
        !(size(m, 2) == size(S, 1) == size(S, 2)) && throw(DimensionMismatch(str))

        any(diag(S) .<= 0) && throw(ArgumentError("The diagonal elements of S must be strictly positive."))
        any(diag(P) .<= 0) && throw(ArgumentError("The diagonal elements of P must be strictly positive."))

        return new(m, P, S)
        end
end

Priors(m::FLOATVEC, P::FLOATMAT, S::Real) = Priors(reshape(m, (length(m), 1)), P, fill(S, 1, 1))
Priors(m::FLOATMAT) = Priors(m, Matrix(Diagonal(ones(size(m, 1)))), Matrix(Diagonal(ones(size(m, 2)))))
Priors(m::FLOATVEC) = Priors(m, Matrix(Diagonal(ones(size(m, 1)))), fill(S, 1, 1))

mutable struct Hyperparameters
    β::Real
    δ::Real
    ν::Real
    function Hyperparameters(β::Real, δ::Real)
        (δ > 0   && δ <= 1) || throw(ArgumentError("0 < δ ≤ 1 required (currently $δ)."))
        (β > 2/3 && β  < 1) || throw(ArgumentError("$(2//3) < β < 1 required (currently $β)."))
        n = 1/(1 - β)
        ν = n * β
        return new(β, δ, ν)
    end
end
Hyperparameters() = Hyperparameters(0.99, 0.99)

mutable struct TVVAR <: TimeVaryingModels
    y::FLOATMAT
    priors::Priors
    hyperparam::Hyperparameters
    m::FLOATARR
    P::FLOATARR
    S::FLOATARR
    Q::FLOATVEC
    μ::FLOATMAT
    Σ::FLOATARR
    e::FLOATMAT
    u::FLOATMAT
    loglik::FLOATVEC
    T::Integer
    p::Integer
    d::Integer
    k::Real

    function TVVAR(y::FLOATMAT, priors::Priors, hyperparam::Hyperparameters)

        T, p = size(y)
        d = Integer((size(priors.m, 1) - 1)/p)

        μ = zeros(T, p)
        Σ = zeros(T, p, p)
        e = zeros(T, p)
        u = zeros(T, p)
        loglik = zeros(T)

        m = zeros(T, d*p + 1, p) ; m[1, :, :] = priors.m
        P = zeros(T, d*p + 1, d*p + 1) ; P[1, :, :] = priors.P
        S = zeros(T, p, p) ; S[1, :, :] = priors.S
        Q = zeros(T)

        k = (hyperparam.β - p*hyperparam.β + p)/(2hyperparam.β - p*hyperparam.β + p - 1)

        return new(y, priors, hyperparam, m, P, S, Q, μ, Σ, e, u, loglik, T, p, d, k)
    end
end