module TVVAR

# Code based on Forecasting with time-varying vector autoregressive models (2008), K. Triantafyllopoulos

# Import
using LinearAlgebra #: diag, kron, I
using Distributions #: Normal, MvNormal
using Plots

# Include scripts
include("./src/types.jl")
include("./src/utils.jl")
include("./src/estimate.jl")
include("./src/forecast.jl")
include("./src/simulate.jl")

include("./example/univariate.jl")

# Exported types
export StateSpace, LocalLevel, LocalLevelTrend

# Exported functions
export estimate

end