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
    @test RootNode(randn(Float32, 10, 3)).dim == 3
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

    @test all(isapprox.(n1.input, output!(n1)))    
    @test n1.level == 0

end

@testset "Nodes" begin
    #test basic properties
    r1 = RootNode{Float32}(3)
    node1 = Node{Float32}(5, r1, activation=:identity)
    @test node1.inpdim == 3
    @test size(node1.weights) == (5,3)
    @test node1.level == 1
    #check datatype inheritance
    node2 = Node(5, r1)
    @test isa(eltype(node2.weights), Type{Float32})

    #test bias
    input = zeros(Float32, 3)
    output = apply!(node1, input)
    @test size(output) == (5,)
    @test all(isapprox.(output, node1.bias.data))

    #test output and input caching
    node2 = Node{Float32}(4, r1, activation=:identity)
    input2 = ones(Float32, 3)
    node2.weights.data .= ones(4,3)
    node2.bias.data .= ones(4)
    s = computelinear!(node2, input2)
    @test all(isapprox.(node2._cachedinput, input2))
    @test all(isapprox.(s, ones(4) * 4))
    
    input3 = ones(Float32, 20, 3)
    s2 = computelinear!(node2, input3)

    y = apply!(node2, input2)
    @test all(isapprox.(y, s))
    @test all(isapprox.(node2._cachedvectorjac, ones(4)))

    #check that levels add properly
    node3 = Node{Float32}(2, node2, activation=:cos)
    @test node3.level == 2

    #check ancestors
    anc = getancestors(node3)
    @test length(anc) == 1
    @test anc[1] === node2

end

@testset "Reduce nodes" begin
    r1 = RootNode(ones(5, 1))
    n1 = Node(2, r1, init=:ones)
    s = SumNode(n1)
    @test s.level == 2
    out = apply!(s, ones(5, 2))
    @test isapprox(out, ones(5) * 2)
end


@testset "Node passes" begin
    r1 = RootNode(rand(10, 3))
    n1 = Node(4, r1)
    n2 = Node(5, n1)
    y2 = forward!(n2)
    @test size(y2) == (10, 5)

    #test node backward pass
    r1 = RootNode(ones(5, 2))
    n1 = Node(3, r1, activation=:sin, init=:ones)
    y = forward!(n1)
    upgrads = ones(5, 3)
    biasgrad, wtgrad, inpgrad = getlocalbatchgrads(n1, upgrads)
    @test size(biasgrad) == (5, 3)
    @test size(wtgrad) == (5, 3, 2)
    @test size(inpgrad) == (5, 2)

    @test isapprox(biasgrad, cos.(ones(5, 3) * 3))
    @test isapprox(wtgrad, ones(5, 3, 2) * cos(3))
    @test isapprox(inpgrad, ones(5, 2) * 3 * cos(3))

    inpgrad = backward!(n1, upgrads)
    @test isapprox(inpgrad, ones(5, 2) * 3 * cos(3))
    @test isapprox(n1.weights.grad, ones(3, 2) * cos(3))
    @test isapprox(n1.bias.grad, ones(3) * cos(3))
end

@testset "Graph" begin
    g = Graph{Float32}()
    @test numroots(g) == 0
    @test numlayers(g) == 0
    @test numoutputs(g) == 0

    rn1 = RootNode{Float32}(1)
    n1 = Node{Float32}(2, rn1)
    n2 = Node{Float32}(1, rn1)
    n3 = Node{Float32}(3, n2)
    addnode!(g, rn1)
    @test numroots(g) == 1
    addnode!(g, n1)
    @test numlayers(g) == 1
    @test length(g.nodes[1]) == 1
    addnode!(g, n2)
    addnode!(g, n3)
    @test numlayers(g) == 2
    @test numoutputs(g) == 1
    @test length(g.nodes[1]) == 2
    @test n1._layerindex == 1
    @test n2._layerindex == 2
    @test n3._layerindex == 1

    #nodes without intermediate layers should throw exception
    n4 = Node{Float32}(2, n3)
    n5 = Node{Float32}(2, n4)
    @test_throws ArgumentError addnode!(g, n5)

    #test batchsize
    rn1 = RootNode(randn(Float32, 10, 4))
    g = Graph{Float32}()
    addnode!(g, rn1)
    @test batchsize(g) == 10
end

@testset "Graph forward pass" begin
    g = Graph{Float32}()
    rn = RootNode(ones(Float32, 1))
    n1 = Node(1, rn, init=:ones)
    n2 = Node(1, n1, init=:ones, activation=:tanh)
    addnode!(g, rn)
    addnode!(g, n1)
    addnode!(g, n2)
    outputs = forward!(g)
   
    @test length(outputs) == 1
    #check that cached results are as expected
    @test isapprox(n1._cachedinput, ones(1))
    @test isapprox(n2._cachedinput, ones(1)*2)
    @test isapprox(n1._cachedvectorjac, ones(1))

    #check that output is as expected
    @test isapprox(outputs[1], tanh.(ones(1)*3))
end

@testset "Graph backward pass" begin
    g = Graph{Float32}()
    rn = RootNode(ones(Float32, 5, 2)*2)
    n1 = Node(2, rn, init=:ones)
    n2 = Node(1, n1, init=:ones, activation=:sin)
    out = SumNode(n2)
    for node in (rn, n1, n2, out)
        addnode!(g, node)
    end
    outputs=forward!(g)
    backward!(g)
    @test isapprox(n2.bias.grad, ones(1) * cos(11))
    @test isapprox(n2.weights.grad, ones(1, 2) * 5 * cos(11))
    @test isapprox(n1.bias.grad, ones(2) * cos(11))
    @test isapprox(n1.weights.grad, ones(2, 2) * cos(11) * 2)
    
end


@testset "miscellaneous" begin
    x = ones(3)
    w = Variable(ones(2, 3))
    b = Variable(ones(2))
    y = applylinear(w, b, x)
    @test all(isapprox.(y, ones(2) * 4))

    x2 = ones(10, 3)
    y2 = applylinear(w, b, x2)
    @test all(isapprox.(y2, ones(10, 2) * 4))

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

    #test initializers
    init = getinitializer(:zeros)
    x = init(Float32, 3, 2)
    @test size(x) == (3,2)
    @test isapprox(sum(x), 0)

    init = getinitializer(:ones)
    x = init(Float32, 4)
    @test isapprox(sum(x), 4)

    #test batch combination
    c = getcombiner(:mean)
    a = ones(3, 4)
    @test isapprox(c(a), ones(4))
end