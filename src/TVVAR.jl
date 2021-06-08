#module TVVAR

# Code based on Forecasting with time-varying vector autoregressive models (2008), K. Triantafyllopoulos

# Import
using LinearAlgebra: diag, kron, I, cholesky
using Distributions: Normal, MvNormal

# Include scripts
include("./src/types.jl")
include("./src/utils.jl")
include("./src/estimate.jl")

include("./example/univariate.jl")

# Exported types
export TVVAR, KF

# Exported functions
export estimate

#end