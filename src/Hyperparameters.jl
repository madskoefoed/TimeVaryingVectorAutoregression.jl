mutable struct Hyperparameters
    β::AbstractFloat
    δ::AbstractFloat
    ν::AbstractFloat

    function Hyperparameters(β::AbstractFloat, δ::AbstractFloat)
        (δ > 0   && δ <= 1) || throw(ArgumentError("0 < δ ≤ 1 required (currently $δ)."))
        (β > 2/3 && β  < 1) || throw(ArgumentError("$(2//3) < β < 1 required (currently $β)."))
        n = 1/(1 - β)
        ν = n * β
        
        return new(β, δ, ν)
    end
end

Hyperparameters() = Hyperparameters(0.99, 0.99)