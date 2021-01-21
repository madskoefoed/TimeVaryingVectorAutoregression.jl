"""
Example: a univariate time series (local linear trend) with an increasing variance, going from 1 to 5.
"""

T = 1_000
y = zeros(T, 1)
F = [1, 0]
G = Matrix(I, 2, 2)
m = zeros(2, 1)
P = Matrix(I, 2, 2) * 1000
S = Matrix(I, 1, 1)

#s = StateSpaceModel(y, F, G, m, P, S, 0.95, 0.95)
s = LocalLevelTrend(y, m, P, S, 0.98, 0.98)


# Simulate
simulate!(s)

# Estimate
m = estimate(s)

plot([s.y m.predicted.μ m.filtered.μ], labels = ["Simulated" "Predicted" "Filtered"], legend = :topleft)

#plot([y m.predicted.μ m.filtered.μ], labels = ["True" "Predicted" "Filtered"], legend = :topleft)
#plot([fill(Σ, T) m.predicted.Σ[:, 1, 1] m.filtered.Σ[:, 1, 1]][2:end, :], labels = ["True" "Predicted" "Filtered"], legend = :topleft)