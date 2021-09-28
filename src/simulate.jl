function simulate(β::FLOATMAT, Σ::FLOATMAT, T = 1_000)
    @assert size(β, 2) == size(Σ, 1) == size(Σ, 2) "The dimensions of β and Σ do not match."
    a, p = size(β)
    #if a == 1
    #    return simulate(β[1, :], Σ, T)
    #end
    d = convert(Int, (a - 1)/2)
    y = zeros(T+d, p)
    #y[1:d, :] = simulate(β[1, :], Σ, d)
    F = ones(a)
    for t in d+1:T+d
        F[2:end] = vec(y[t-1:-1:t-d, :])
        y[t, :] = rand(MvNormal(vec(F' * β), Σ))
    end
    return y
end

#function simulate(β::FLOATVEC, Σ::FLOATMAT, T = 1_000)
#    @assert length(β) == size(Σ, 1) == size(Σ, 2) "The dimensions of β and Σ do not match."
#   y = Matrix(rand(MvNormal(β, Σ), T)')
#    return y
#end

function simulate(model::Model)
    y, m, P, S, β, δ = model.y, model.m, model.P, model.S, model.β, model.δ
    
    T, p = size(y)
    a = size(m, 1)
    d = convert(Int, (a - 1)/2)
    y = zeros(T+d, p)
    F = ones(a)

    sim = Simulation(zeros(T, p), zeros(T, a, p))

    for t in d+1:T
        F[2:end] = vec(y[t-1:-1:t-d, :])
        # Simulate Ω
        sim.m[t, :, :] = rand(MatrixNormal(m, P, S))
        # Simulate y
        sim.y[t, :] = rand(MvNormal(vec(F' * sim.m[t, :, :]), S))
    end
    return sim

end