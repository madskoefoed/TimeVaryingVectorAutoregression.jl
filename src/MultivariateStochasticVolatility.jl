module MultivariateStochasticVolatility

# Code based on Forecasting with time-varying vector autoregressive models (2008), K. Triantafyllopoulos

# Constant
const FLOATVEC = Array{Fl, 1} where Fl <: AbstractFloat
const FLOATMAT = Array{Fl, 2} where Fl <: AbstractFloat

# Import
using LinearAlgebra: diag, kron, I, cholesky, Diagonal
using Distributions: Normal, MvNormal, MvTDist, logpdf
using PDMats: PDMat

# Include scripts
include("Hyperparameters.jl")
include("Priors.jl")
include("Posteriors.jl")
include("Models.jl")
include("Estimate.jl")
include("Simulate.jl")

# Exported types
export Hyperparameters
export Priors, PriorPredictive
export Posteriors, PosteriorPredictive

# Exported functions
export estimate, simulate

end