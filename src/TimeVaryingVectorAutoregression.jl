module TimeVaryingVectorAutoregression

# Code based on Forecasting with time-varying vector autoregressive models (2008), K. Triantafyllopoulos

# Constant
const FLOATVEC = Array{Fl, 1} where Fl <: AbstractFloat
const FLOATMAT = Array{Fl, 2} where Fl <: AbstractFloat
const FLOATARR = Array{Fl, 3} where Fl <: AbstractFloat

# Import
using LinearAlgebra: diag, kron, I, cholesky, Diagonal
using Distributions: Normal, MvNormal, MvTDist, logpdf
using PDMats: PDMat

# Include scripts
include("types.jl")
include("estimate.jl")

# Exported types
export TVVAR

# Exported functions
export estimate_batch, predict!, update!

end