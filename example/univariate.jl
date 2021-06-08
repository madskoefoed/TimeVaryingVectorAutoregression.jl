"""
Example: a AR(2) time series with 1000 observations. The variance quadruples after 500 observations.
"""

# Packages
using Plots
using Distributions

# Generate an AR(2) time series with a doubling in standard deviation after half the observations observations:
# y(t) = -1.0 + 0.5 * y(t-1) - 0.5 * y(t- 2) + Σ
α = -1.0
β = [0.5, -0.5]
b = length(β)
T = 1_000
y = zeros(T, 1);
Σ = ones(T)
for t in 1:T
    if t <= b
        y[t] = α + rand(Normal(0.0, 1.0))
    else
        if t <= T/2
            # Initial volatility (1.0)
            y[t] = α + y[t-1:-1:t-b]' * β + rand(Normal(0.0, 1.0))
        else
            # Double volatility (2.0)
            Σ[t] = 4.0
            y[t] = α + y[t-1:-1:t-b]' * β + rand(Normal(0.0, sqrt(Σ[t])))
        end
    end
end

# Construct priors
priors = TVVAR(y, fill(0.0, b+1, 1), Matrix(1000.0I, b+1, b+1), Matrix(1.0I, 1, 1), 0.95, 0.95)

# Estimate
est = estimate(priors)

# Plot simulated data and estimated means
pl = scatter(est.y, color = :blue, markeralpha = 0.5, label = "observed", title = "Time series: AR(2)", legend = :topleft)
pl = plot!(est.μ, color = :blue, linewidth = 2; label = "predicted")

savefig(pl,"./example/timeseries_uni.png")

# Plot the estimated vs. true coefficients
cols = [:blue :red :green :yellow]
pl = plot(est.m[:, :, 1], ylim = [-2, 2], color = cols, label = "", title = "Coefficients")
pl = plot!(fill(α, T), label = "a", color = cols[1], linestyle = :dash)
pl = plot!(fill(β[2], T), label = "b(1)", color = cols[2], linestyle = :dash)
pl = plot!(fill(β[1], T), label = "b(2)", color = cols[3], linestyle = :dash)

savefig(pl,"./example/coefficients_uni.png")

# Plot true and estimated variances
pl = plot(est.Σ[:, 1, 1], color = :blue, linewidth = 1, label = "", ylim = [0.0, 10.0], title = "Variance")
pl = plot!(Σ, color = :blue, linestyle = :dash, label = "")

savefig(pl,"./example/variance_uni.png")