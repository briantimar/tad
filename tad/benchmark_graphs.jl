include("graphs.jl")

using BenchmarkTools

println("Starting graphs benchmarks...")
rn = RootNode{Float64}(10, rand(1000,10))
println(@benchmark output(rn))