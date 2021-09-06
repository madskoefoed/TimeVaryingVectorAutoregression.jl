"""
Example: a VAR(2, 2) time series with 1000 observations. The variance quadruples after 500 observations.
"""

# Packages
using Plots
using Distributions

# Generate an VAR(2, 1) time series. The two series are positively correlated:
# y(t) = [-1.0 1.0] + [0.5 -0.5] * y(t-1) + Σ
α = [-1.0 1.0]
β = [0.5 0.0; 0.0 -0.5]
b = convert(Int, size(β, 1)/2)
T = 1_000
y = zeros(T, 2);
Σ = Matrix(cholesky([1.0 0.5; 0.5 4.0]).L)
for t in 1:T
    if t <= b
        y[t, :] = α + rand(MvNormal([0.0, 0.0], I))' * Σ
    else
        y[t, :] = α + vec(y[t-1:-1:t-b, :])' * β + rand(MvNormal([0.0, 0.0], I))' * Σ
    end
end

# Construct priors
model = Model(y, fill(0.0, 3, 2), Matrix(1000.0I, 3, 3), Matrix(1.0I, 2, 2), 0.99, 0.99)

# Estimate
out = estimate(model)

# Plot simulated data and estimated means
pl = scatter(est.y, color = [:blue :red], markeralpha = 0.5, label = ["observed y(1)" "observed y(2)"], title = "Time series: VAR(2, 1)", legend = :bottom, ylim = [-10.0, 10.0])
pl = plot!(est.μ, color = [:blue :red], linewidth = 2; label = ["predicted y(1)" "predicted y(2)"])

savefig(pl,"./example/timeseries_multi.png")

# Plot the estimated vs. true coefficients
cols = [:blue :red :green :yellow :pink :orange]
pl = plot(est.m[:, :, 1], ylim = [-2, 2], color = cols[:, 1:3], label = "", title = "Coefficients", legend = :top)
pl = plot!(est.m[:, :, 2], color = cols[:, 4:6], label = "")
pl = plot!(repeat(α, T, 1), label = ["a(1)" "a(2)"], color = cols[:, [1, 4]], linestyle = :dash)
pl = plot!(repeat(β[2, :]', T, 1), label = ["b(1, 1)" "b(1, 2)"], color = cols[:, [3, 6]], linestyle = :dash)
pl = plot!(repeat(β[1, :]', T, 1), label = ["b(2, 1)" "b(2, 2)"], color = cols[:, [2, 5]], linestyle = :dash)

savefig(pl,"./example/coefficients_multi.png")

# Plot true and estimated variances
pl = plot(hcat(est.Σ[:, 1, 1], est.Σ[:, 2, 2]), color = [:blue :red], linewidth = 1, label = "", ylim = [0.0, 10.0], title = "Variance")
pl = plot!(Σ, color = :blue, linestyle = :dash, label = "")

savefig(pl,"./example/variance_multi.png")