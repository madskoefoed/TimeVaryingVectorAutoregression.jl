"""
Example: a univariate time series (local linear trend) with an increasing variance, going from 1 to 5.
"""

# Packages
using Plots
using Distributions

# Generate an AR(3) time series with a doubling in volatility after half the observations observations:
# y(t) = -1.0 + 0.5 * y(t-1) - 0.2 y(t-2) - 0.3 * y(t- 3) + Σ
α = -1.0
β = [0.5, 0.0, -0.5]
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
priors = TVVAR(y, fill(0.0, b+1, 1), Matrix(1000.0I, b+1, b+1), Matrix(1.0I, 1, 1), 0.99, 0.99)

# Estimate
est = estimate(priors)

# Plot simulated data and estimated means
scatter(est.y, color = [:blue], markeralpha = 0.5, label = "observed")
plot!(est.μ, color = [:blue], linewidth = 2; label = "predicted")

# Plot the estimated vs. true coefficients
cols = [:blue :red :green :yellow]
plot(est.m[:, :, 1], ylim = [-2, 2], color = cols, label = "", title = "Coefficients")
plot!(fill(α, T), label = "a", color = cols[1], linestyle = :dash)
plot!(fill(β[3], T), label = "b(1)", color = cols[2], linestyle = :dash)
plot!(fill(β[2], T), label = "b(2)", color = cols[3], linestyle = :dash)
plot!(fill(β[1], T), label = "b(3)", color = cols[4], linestyle = :dash)

# Plot true and estimated variances
plot(sqrt.(est.Σ[:, 1, 1]), color = [:blue], linewidth = 1,
label = "", ylim = [0.0, 5.0], title = "Standard deviation")
plot!(sqrt.(Σ), color = :blue, linestyle = :dash, label = "")

figpath = "C:/Users/brett/OneDrive/Documents/Soc323/"
savefig(fig1,figpath*"fig1.png")
savefig(fig1,figpath*"fig1.svg")
savefig(fig1,figpath*"fig1.pdf")

# Construct priors
priors2 = MSV(y, Matrix(1.0I, 1, 1), 0.99)

# Estimate
est2 = estimate(priors2)