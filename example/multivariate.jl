"""
Example: a VAR(2, 2) time series with 1000 observations. The variance quadruples after 500 observations.
"""

# Packages
using Plots
using Distributions

# Generate an VAR(2) time series. The two series are negatively correlated:
# y(t) = [-1.0 1.0] + [0.5 -0.5] * y(t-1) + Σ
#β = [-1.0 1.0; 0.5 0.0; -0.5 0.0; 0.0 -0.5; 0.0 0.0]
#β = [-1.0 1.0; 0.5 0.0; 0.0 -0.5]
β = [-1.0 1.0;]
Σ = [1.0 -0.5; -0.5 2.0]
y = simulate(β, Σ, 250)

# Construct priors
#model = Model(y, fill(0.0, 5, 2), Matrix(1000.0I, 5, 5), Matrix(1.0I, 2, 2), 0.99, 0.99)
#model = Model(y, fill(0.0, 3, 2), Matrix(1000.0I, 3, 3), Matrix(1.0I, 2, 2), 0.99, 0.99)
model = Model(y, fill(0.0, 1, 2), Matrix(1.0I, 1, 1), Matrix(1.0I, 2, 2), 0.99, 0.99)

# Estimate
out = estimate(model)

# Plot simulated data and estimated means
pl = scatter(out.y, color = [:blue :red], markeralpha = 0.5, label = ["observed y(1)" "observed y(2)"], title = "Time series: VAR(2)", legend = :bottomright, ylim = [-8.0, 8.0])
pl = plot!(out.μ, color = [:blue :red], linewidth = 2; label = ["predicted y(1)" "predicted y(2)"])

savefig(pl,"./example/timeseries_multi.png")

# Plot the estimated vs. true coefficients for y(1)
cols = [:blue :red :green :yellow :grey]
pl = plot(out.m[:, :, 1], ylim = [-2, 2], color = cols, label = "", title = "Coefficients for y(1)", legend = :top)
pl = plot!(repeat(β[:, 1]', size(y, 1), 1), label = ["a" "b(1)" "b(2)" "b(3)" "b(4)"], color = cols, linestyle = :dash)

savefig(pl,"./example/coefficients_multi_y1.png")

# Plot the estimated vs. true coefficients for y(2)
pl = plot(out.m[:, :, 2], ylim = [-2, 4], color = cols, label = "", title = "Coefficients for y(2)", legend = :top)
pl = plot!(repeat(β[:, 2]', size(y, 1), 1), label = ["a" "b(1)" "b(2)" "b(3)" "b(4)"], color = cols, linestyle = :dash)
savefig(pl,"./example/coefficients_multi_y2.png")

# Plot true and estimated variances
pl = plot(hcat(out.Σ[:, 1, 1], out.Σ[:, 2, 2]), color = cols, linewidth = 1, label = ["Var(1)" "Var(2)"], ylim = [0.0, 10.0], title = "Variance")
pl = hline!([Σ[1, 1] Σ[2, 2]], color = cols, linestyle = :dash, label = "")

savefig(pl,"./example/variance_multi.png")