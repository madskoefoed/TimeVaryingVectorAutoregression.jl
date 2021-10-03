
function simulate!(sim::Simulation)

    T, p = size(sim.y)
    a = size(sim.m, 1)
    d = convert(Int, (a - 1)/2)
    y = zeros(T+d, p)
    F = ones(a)

    for t in 1:T
        if t > d
            # Lagged values
            F[2:end] = vec(sim.y[t-1:-1:t-d, :])
            # Simulate Ω
            sim.Ω[t, :, :] = rand(MatrixNormal(sim.m, sim.P, sim.S))
            # Simulate y
            sim.y[t, :] = rand(MvNormal(vec(F' * sim.Ω[t, :, :]), sim.S))
        else
            sim.Ω[t, :, :] = sim.m
        end
    end
    #return sim
end