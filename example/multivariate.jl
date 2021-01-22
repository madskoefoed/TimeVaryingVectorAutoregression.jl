
# Packages
using Plots
using Distributions

# Generate time series with a doubling in volatility after 500 observations
Σ = cholesky([2 -1; -1 4])
T = 500
y = randn(T, 2) * Σ.U
Σ = cholesky([1 0; 0 1])
T = 500
y = vcat(y, randn(T, 2) * Σ.U) .+ [1.0 -4.0]

# Construct priors
priors = Priors([-5.0 2.0], Matrix(1000.0I, 1, 1), Matrix(1.0I, 2, 2), 0.95, 0.95)

# Simulate
#O = simulate!(s)

# Estimate
estimated = estimate(y, priors)

# Plot simulated data and estimated means
scatter(estimated.y, color = [:blue :red], markeralpha = 0.5, layout = (2, 1), label = "y")
plot!(estimated.μ, color = [:blue :red], linewidth = 2; label = "μ")

# Plot true and estimated variances
plot([estimated.Σ[:, 1, 1] estimated.Σ[:, 2, 2]], color = [:blue :red], linewidth = 1; label = "Predicted variance", layout = (2, 1), ylim = [0.0, 10.0])
plot!([[ones(T)*2;ones(T)] [ones(T)*4;ones(T)]], color = :grey, linewidth = 2, label = "True variance")

# Plot estimated variances vs. variance of state
plot([estimated.Σ[:, 1, 1] estimated.Σ[:, 2, 2]], color = [:blue :red], linewidth = 1; label = "", layout = (2, 1), ylim = [0.0, 10.0])
plot!([estimated.Ω[:, 1, 1] estimated.Ω[:, 2, 2]], color = :grey, label = "")