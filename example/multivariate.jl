"""
Example: a VAR(2, 2) time series with 1000 observations. The variance quadruples after 500 observations.
"""

# Packages
using Plots
using Distributions

# Generate an VAR(2) time series. The two series are positively correlated.
β = [-1.0  1.0;
      0.5  0.0;
     -0.5  0.1;
      0.0 -0.5;
      0.3  0.3];
Σ = [1.0 0.5;
     0.5 2.0];
y = simulate(β, Σ, 500);

sim = Simulation([ 1.0 -1.0;
                  -0.4  0.4;
                   0.2 -0.2],
                   Matrix(0.1*I, 3, 3),
                   Matrix(0.5*I, 2, 2),
                   100);
y = simulate(model)

# Construct priors
model = Model(y, fill(0.0, 5, 2), Matrix(1000.0I, 5, 5), Matrix(100.0I, 2, 2), 0.99, 0.99);

# Estimate
out = estimate(model);

# Plot simulated data and estimated means
pl = scatter(out.y, color = [:blue :red], markeralpha = 0.5, label = ["observed y₁" "observed y₂"], title = "Time series: VAR(2)", legend = :bottomright, ylim = [-8.0, 8.0])
pl = plot!(out.μ, color = [:blue :red], linewidth = 2; label = ["predicted y₁" "predicted y₂"])

savefig(pl,"./example/timeseries_multi.png")

# Plot the estimated vs. true coefficients for y(1)
cols = [:blue :red :green :yellow :grey];
lgds = ["int" "b₁" "b₂" "b₃" "b₄"];
pl = plot(out.m[:, :, 1], ylim = [-1.5, 1.5], color = cols, label = "", title = "Coefficients for y₁", legend = :topright)
pl = plot!(repeat(β[:, 1]', size(y, 1), 1), label = lgds, color = cols, linestyle = :dash)

savefig(pl,"./example/coefficients_multi_y1.png")

# Plot the estimated vs. true coefficients for y(2)
pl = plot(out.m[:, :, 2], ylim = [-2, 2], color = cols, label = "", title = "Coefficients for y₂", legend = :bottom)
pl = plot!(repeat(β[:, 2]', size(y, 1), 1), label = lgds, color = cols, linestyle = :dash)

savefig(pl,"./example/coefficients_multi_y2.png")

# Plot true and estimated variances
pl = plot(hcat(out.Σ[:, 1, 1], out.Σ[:, 2, 2], out.Σ[:, 1, 2]), color = cols, linewidth = 1, label = ["Var(y₁)" "Var(y₂)" "Cov(y₁, y₂)"], ylim = [0, 3], title = "Covariances")
pl = hline!([Σ[1, 1] Σ[2, 2] Σ[1, 2]], color = cols, linestyle = :dash, label = "")

savefig(pl,"./example/variance_multi.png")