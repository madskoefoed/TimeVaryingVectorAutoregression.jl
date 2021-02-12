"""
Example: a univariate time series (local linear trend) with an increasing variance, going from 1 to 5.
"""

# Packages
using Plots
using Distributions

# Generate time series with a doubling in volatility after half the observations observations
T = 1_000
Σ = vcat(fill(1.0, convert(Integer, T/2)), fill(2.0, convert(Integer, T/2)))
y = randn(T, 1) .* sqrt.(Σ) .+ 5.0

# Construct priors
tvvar_priors = TVVAR(y, fill(-1.0, 1, 1), Matrix(1000.0I, 1, 1), Matrix(1.0I, 1, 1), 0.99, 0.99)
kf_priors    = TVVAR(y, fill(-1.0, 1, 1), Matrix(1000.0I, 1, 1), Matrix(1.0I, 1, 1), 0.99)

# Estimate
tvvar_est = estimate(tvvar_priors)
kf_est    = estimate(kf_priors)

# Plot simulated data and estimated means
scatter(tvvar_est.y, color = [:blue], markeralpha = 0.5, label = "y")
plot!(tvvar_est.μ, color = [:blue], linewidth = 2; label = "μ (TVVAR)")
plot!(kf_est.μ, color = [:blue], linewidth = 2; label = "μ (KF)")

# Plot true and estimated variances
plot(tvvar_est.Σ[:, 1, 1], color = [:blue], linewidth = 1; label = "Predicted variance", ylim = [0.0, 10.0])
plot!(Σ, color = :blue, linewidth = 2, label = "True variance")

# Plot estimated variances vs. variance of state
plot(tvvar_est.Σ[:, 1, 1], color = [:blue], linewidth = 1; label = "System variance", ylim = [0.0, 10.0])
plot!(tvvar_est.Ω[:, 1, 1], color = :red, label = "State variance")