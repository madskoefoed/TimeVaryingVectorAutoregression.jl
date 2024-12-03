using MultivariateStochasticVolatility
using Test
using PDMats
using LinearAlgebra

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
    h = Hyperparameters()
    p = 2
    m = [-1.0, 5.0]
    P = 999.9
    S = PDMat(Matrix(Diagonal(ones(p)*10)))
    priors = Priors(m, P, S, h)
    @test length(priors.m) == p
    @test length(priors.P) == 1
    @test size(priors.S, 1) == size(priors.S, 2)
end

################
### Simulate ###
################
@testset "Simulate" begin
    T = 50
    p = 3
    Φ = rand(p)
    Σ = PDMat(Matrix(Diagonal([1, 2, 3])))
    y = simulate(T, Φ, Σ)

    @test size(y, 1) == T
    @test size(y, 2) == p
end

################
### Estimate ###
################
#@testset "Estimate" begin
#end

