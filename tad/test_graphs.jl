using Test

include("graphs.jl")

@testset "Variables" begin
    data = randn(Float32, (5,5))
    v1 = Variable(data)
    @test eltype(v1.data) == eltype(data)
    @test isapprox(sum(v1.grad), 0)
    @test isapprox(v1[1, 1], v1.data[1,1])
end

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
    node1 = Node{Float32}(5, r1, :identity)
    @test node1.inpdim == 3
    @test size(node1.weights) == (5,3)

    input = zeros(Float32, 3)
    output = apply!(node1, input)
    @test size(output) == (5,)
    @test all(isapprox.(output, node1.bias.data))

    node2 = Node{Float32}(4, r1, :identity)
    input2 = ones(Float32, 3)
    node2.weights.data .= ones(4,3)
    node2.bias.data .= ones(4)
    s = computelinear!(node2, input2)
    @test all(isapprox.(node2._cachedinput, input2))
    @test all(isapprox.(s, ones(4) * 4))
    
    y = apply!(node2, input2)
    @test all(isapprox.(y, s))
    @test all(isapprox.(node2._cachedvectorjac, ones(4)))


end

@testset "miscellaneous" begin
    x = ones(3)
    w = Variable(ones(2, 3))
    b = Variable(ones(2))
    y = applylinear(w, b, x)
    @test all(isapprox.(y, ones(2) * 4))

    #test act fns
    x = zeros(4)
    y1 = getactivation(:identity).(x)
    @test all(isapprox.(y1, x))
    y2 = getactivation(:cos).(x)
    @test all(isapprox.(y2, ones(4)))

    #test vector jacs
    x = zeros(4)
    j1 = getgradient(:identity).(x)
    @test all(isapprox.(j1, ones(4)))
    j2 = getgradient(:sin).(x)
    @test all(isapprox.(j2, ones(4)))
    x2 = rand(4)
    j3 = getgradient(:tanh).(x2)
    @test all(isapprox(j3, 1 .- tanh.(x2).^2))
end