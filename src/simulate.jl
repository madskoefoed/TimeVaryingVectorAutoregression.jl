function simulate(β::FLOATMAT, Σ::FLOATMAT, T = 1_000)
    @assert size(β, 2) == size(Σ, 1) == size(Σ, 2) "The dimensions of β and Σ do not match."
    a, p = size(β)
    if a == 1
        return simulate(β[1, :], Σ, T)
    end
    d = convert(Int, (a - 1)/2)
    y = zeros(T+d, p)
    y[1:d, :] = simulate(β[1, :], Σ, d)
    F = ones(a)
    for t in d+1:T+d
        F[2:end] = vec(y[t-1:-1:t-d, :])
        y[t, :] = rand(MvNormal(vec(F' * β), Σ))
    end
    return y[d+1:end, :]
end

function simulate(β::FLOATVEC, Σ::FLOATMAT, T = 1_000)
    @assert length(β) == size(Σ, 1) == size(Σ, 2) "The dimensions of β and Σ do not match."
    y = Matrix(rand(MvNormal(β, Σ), T)')
    return y
end