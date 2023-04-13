# Packages
using Plots

# Construct data
T_example = 100

μ_example = [ 1.0 -3.0;
              0.5  0.0;
             -0.2 -0.8;
             -0.2  0.0;
              0.0  0.5;
              0.0  0.0;
              0.2  0.3]

p_example = size(μ_example, 2)
d_example = Integer((size(μ_example, 1)-1)/p_example)

y_example = zeros(T_example, p_example)
for t in (d_example+1):T_example
    F_example = ones(d_example*p_example+1, 1)
    F_example[2:end] = vec(y_example[t-d_example:t-1, :])
    m_example = F_example' * μ_example
    e_example = rand(MvNormal([0.5 0.0; 0.0 0.25]))
    y_example[t, :] = m_example .+ e_example'
end

# Priors
priors_example = Priors(zeros(d_example*p_example+1, p_example),
                        Matrix(Diagonal(ones(d_example*p_example+1)*1000)),
                        Matrix(Diagonal(ones(p_example))));

hyperparam_example = Hyperparameters(0.99, 0.99);

tvvar1 = TVVAR(y_example, priors_example, hyperparam_example);

estimate_batch!(tvvar1)

tvvar2 = TVVAR(y_example[1:T_example-20, :], priors_example, hyperparam_example);

estimate_batch!(tvvar2)

for t in T_example-19:T_example
    println("Time: $t")
    predict!(tvvar2)
    update!(tvvar2, y_example[t, :])
end

plot(tvvar2.μ, lw=1, lc=[:black :red], label="")
scatter!(y_example, lw=3, mc=[:black :red], ma=0.5, label=["y1" "y2"])
plot!(legend=:outerbottom, legendcolumn=2)