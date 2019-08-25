using Test

@testset "test graphs.jl" begin
    include("graphs.jl")
    @test RootNode{Float32}(4).dim == 4
    @test isa(gettype(RootNode{Float32}(3)), Type{Float32})
end