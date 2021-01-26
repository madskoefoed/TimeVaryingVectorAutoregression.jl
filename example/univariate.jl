"""
Example: a univariate time series (local linear trend) with an increasing variance, going from 1 to 5.
"""

# Packages
using Plots
using Distributions

# Generate time series with a doubling in volatility after 500 observations
Σ = 2.0
T = 10_000
y = randn(T, 1) * Σ

# Construct priors
priors = Priors(fill(-1.0, 1, 1), Matrix(1000.0I, 1, 1), Matrix(1.0I, 1, 1), 0.998, 0.9995)

# Simulate
#O = simulate!(s)

# Estimate
estimated = estimate(y, priors)

# Plot simulated data and estimated means
scatter(estimated.y, color = [:blue], markeralpha = 0.5, label = "y")
plot!(estimated.μ, color = [:blue], linewidth = 2; label = "μ")

# Plot true and estimated variances
plot([estimated.Σ[:, 1, 1] estimated.Σ[:, 2, 2]], color = [:blue :red], linewidth = 1; label = "Predicted variance", layout = (2, 1), ylim = [0.0, 10.0])
plot!([[ones(T)*2;ones(T)] [ones(T)*4;ones(T)]], color = :grey, linewidth = 2, label = "True variance")

# Plot estimated variances vs. variance of state
plot([estimated.Σ[:, 1, 1] estimated.Σ[:, 2, 2]], color = [:blue :red], linewidth = 1; label = "", layout = (2, 1), ylim = [0.0, 10.0])
plot!([estimated.Ω[:, 1, 1] estimated.Ω[:, 2, 2]], color = :grey, label = "")