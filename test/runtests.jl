using TimeVaryingVectorAutoregression
using Test

#######################
### Hyperparameters ###
#######################
@testset "Hyperparameters" begin
    h = Hyperparameters()
    @test isapprox(h.β, 0.99)
    @test isapprox(h.δ, 0.99)

    h = Hyperparameters(0.8, 0.9)
    @test isapprox(h.β, 0.8)
    @test isapprox(h.δ, 0.9)
end

##############
### Priors ###
##############
@testset "Priors" begin
    d = 3
    p = 2
    Pr = Priors(rand(d*p+1,p)/100, get_diag_covmat(d*p+1, 1000.0), get_diag_covmat(p, 1.0))
    @test size(Pr.m, 1) == d*p+1 && size(Pr.m, 2) == p
    @test size(Pr.P, 1) == d*p+1 == size(Pr.P, 2)
    @test size(Pr.S, 1) == p == size(Pr.S, 2)

    d = 5
    p = 1
    Pr = Priors(rand(d*p+1,p)/100, get_diag_covmat(d*p+1, 1000.0), get_diag_covmat(p, 1.0))
    @test size(Pr.m, 1) == d*p+1 && size(Pr.m, 2) == p
    @test size(Pr.P, 1) == d*p+1 == size(Pr.P, 2)
    @test size(Pr.S, 1) == p == size(Pr.S, 2)

    d = 1
    p = 10
    Pr = Priors(rand(d*p+1,p)/100, get_diag_covmat(d*p+1, 1000.0), get_diag_covmat(p, 1.0))
    @test size(Pr.m, 1) == d*p+1 && size(Pr.m, 2) == p
    @test size(Pr.P, 1) == d*p+1 == size(Pr.P, 2)
    @test size(Pr.S, 1) == p == size(Pr.S, 2)

    d = 1
    p = 10
    Pr = Priors(rand(d*p+1,p)/100, get_diag_covmat(d*p+1, 1000.0), get_diag_covmat(p, 1.0))
    @test size(Pr.m, 1) == d*p+1 && size(Pr.m, 2) == p
    @test size(Pr.P, 1) == d*p+1 == size(Pr.P, 2)
    @test size(Pr.S, 1) == p == size(Pr.S, 2)
end

################
### Simulate ###
################
@testset "Simulate" begin
    d = 1
    p = 2
    Pr = Priors(rand(d*p+1,p)/100, get_diag_covmat(d*p+1, 1000.0), get_diag_covmat(p, 1.0))
    Σ = get_diag_covmat(p, 2.0)
    sim = simulate(150, Pr.m, Σ)

    @test isapprox(sum(Σ), p*2)
    @test size(sim, 1) == 150
    @test size(sim, 2) == p
end

################
### Estimate ###
################
@testset "Estimate" begin
    T0 = 1000
    T1 = 500
    d = 1
    p = 2
    Hy = Hyperparameters(0.99, 0.99)
    β = [5 -10; -0.8 0.0; 0.0 0.9]
    m = zeros(3, 2)
    P = get_diag_covmat(d*p+1, 1000.0)
    S = get_diag_covmat(p, 1.0)
    Pr = Priors(m, P, S)
    Σ = get_diag_covmat(p, 0.1)
    y = simulate(T0+T1, β, Σ)
    tvvar = TVVAR(y[1:T0, :], Pr, Hy)

    estimate_batch!(tvvar)
    
    @test size(tvvar.y, 1) == size(tvvar.μ, 1) == size(tvvar.Σ, 1) == size(tvvar.m, 1) == size(tvvar.P, 1) == size(tvvar.S, 1) == T0

    for t in T0+1:T0+T1
        predict!(tvvar)
        update!(tvvar, y[t, :])
    end

    @test size(tvvar.y, 1) == size(tvvar.μ, 1) == size(tvvar.Σ, 1) == size(tvvar.m, 1) == size(tvvar.P, 1) == size(tvvar.S, 1) == T0+T1

end

