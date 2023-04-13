using TVVAR
using Test

@testset "TVVAR.jl" begin
    #######################
    ### Hyperparameters ###
    #######################

    h = Hyperparameters()
    @test isapprox(h.β, 0.99)
    @test isapprox(h.δ, 0.99)

    h = Hyperparameters(0.8, 0.9)
    @test isapprox(h.β, 0.8)
    @test isapprox(h.δ, 0.9)
    
end
