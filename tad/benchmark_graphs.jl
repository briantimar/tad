include("graphs.jl")

using BenchmarkTools
using Statistics

function showbm(bm)
    println("Mean time $(mean(bm.times)) ns")
    println("Memory $(bm.memory) bytes")
end

macro showbm(bm)
    return quote
            showbm($bm)
    end
end

println("Starting graphs benchmarks...")

function rnoutput()
    x = RootNode{Float64}(10, rand(1000,10));
    return @benchmark output($x)
end

@showbm rnoutput()
