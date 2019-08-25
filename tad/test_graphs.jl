using Test

include("graphs.jl")

@testset "Root nodes" begin

    @test RootNode{Float32} <: CompNode{Float32}
    @test RootNode{Float32}(4, nothing).dim == 4
    @test isa(gettype(RootNode{Float32}(3, nothing)), Type{Float32})

    n1 = RootNode{Float32}(3)
    inp0 = rand(Float32, (2,3))
    inp1 = rand(Float32, (4,4))
    inp2 = rand(Float32, (3, 3, 3))
    inp3 = rand(Float32, (3,))

    @test checkvalidinput(n1, inp0)
    @test checkvalidinput(n1, inp3)
    @test_throws ArgumentError RootNode{Float32}(3, inp1)
    @test_throws ArgumentError RootNode{Float32}(3, inp2)

    setinput!(n1, inp0)
    @test all(isapprox.(n1.input, inp0))

    @test all(isapprox.(n1.input, output(n1)))    

end

@testset "Nodes" begin
    r1 = RootNode{Float32}(3)
    r2 = RootNode{Float32}(4)
    node1 = Node{Float32}(5, [r1, r2], :identity)
    @test node1.inpdim == 7
end