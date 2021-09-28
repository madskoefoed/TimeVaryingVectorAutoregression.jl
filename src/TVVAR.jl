#module TVVAR

# Code based on Forecasting with time-varying vector autoregressive models (2008), K. Triantafyllopoulos

# Constant
const FLOATVEC = Array{Fl, 1} where Fl <: AbstractFloat
const FLOATMAT = Array{Fl, 2} where Fl <: AbstractFloat
const FLOATARR = Array{Fl, 3} where Fl <: AbstractFloat

# Import
using LinearAlgebra: diag, kron, I, cholesky
using Distributions: Normal, MvNormal, MatrixNormal

# Include scripts
include("./src/types.jl")
include("./src/utils.jl")
include("./src/estimate.jl")
include("./src/simulate.jl")

# Exported types
#export TVVAR, KF

# Exported functions
#export estimate

#end