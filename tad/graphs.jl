"""Struct representing a node (operation) in a computational graph"""

using DiffRules

@DiffRules.define_diffrule Base.identity(x) = :(1.0)


_DEFAULT_FLOAT = Float32

abstract type GraphElement end
abstract type AbstractNode <: GraphElement end
abstract type CompNode{T<:Real} <: AbstractNode end

# populate a table with some activation functions...
_ACTIVATION_SYMBOLS = [(:Base, :identity), (:Base, :cos), (:Base, :sin), (:Base, :tanh)]

_ACTIVATIONS = Dict{Union{Expr, Symbol}, Function}()
_ACTIVATION_GRADIENTS = Dict{Union{Expr, Symbol}, Function}()

for act in _ACTIVATION_SYMBOLS
    nmspace, actname = act
    _ACTIVATIONS[actname] = eval(actname)
    _ACTIVATION_GRADIENTS[actname] = eval(:(x -> $(DiffRules.diffrule(nmspace, actname, :x))))
end

_INITIALIZER_SYMBOLS = [:randn, :zeros, :ones]
_INITIALIZERS = Dict(init => eval(:( (args...) -> $(init)(args...))) for init in _INITIALIZER_SYMBOLS )

function getinitializer(name::Symbol)
    name in keys((_INITIALIZERS)) || throw(ArgumentError("Invalid intializer $name"))
    return _INITIALIZERS[name]
end

"A variable which holds trainable weights and gradients"
mutable struct Variable{T<:Real, N} <: AbstractArray{T, N}
    data::Array{T, N}
    grad::Array{T, N}
    function Variable{T, N}(data::AbstractArray{T, N}) where T <: Real where N
        grad = zeros(T, size(data))
        return new(data, grad)
    end
end

#outer constructors for the variable object
function Variable{T}(data::Array{T}) where T <: Real
    N = ndims(data)
    return Variable{T, N}(data)
end

function Variable(data::Array)
    T = eltype(data)
    N = ndims(data)
    return Variable{T, N}(data)
end

Base.size(var::Variable{T, N}) where {T,N} = Base.size(var.data)
Base.getindex(var::Variable{T, N}, index::Vararg{Int, N}) where {T, N} = getindex(var.data, index...)

function set!(var::Variable{T, N}, data::Array{T, N}) where {T, N}
    var.data .= data
end

"""Check if array is compatible with given node dim"""
function checkvalidnodedim(input::Array, dim::Int)
    return ndims(input) <= 2 && size(input)[end] == dim
end

"Check if an array can be supplied as input to a node"
function checkvalidinput(node::CompNode, input::Array)
    return checkvalidnodedim(input, node.dim)
end

"Holds inputs to graph.
    `dim`: Int, dimensionality of the node inputs
    `input`: ndims<2 array with last dimension equal to node dim."
mutable struct RootNode{T<:Real} <: CompNode{T} 
    dim::Int
    input::Union{Array{T}, Nothing}
    level::Int
    _layerindex::Int

    function RootNode{T}(dim::Int, input::Union{Array{T}, Nothing}) where T <: Real 
        isnothing(input)  || (! checkvalidnodedim(input, dim) && throw(ArgumentError("Input has invalid shape $(size(input)) for node dim $dim")) );
        level = 0
        _layerindex = 0
        return new(dim, input, level, _layerindex)
    end

end

# RootNode(dim, input) = RootNode{_DEFAULT_FLOAT}(dim, input)
RootNode{T}(dim::Int) where T = RootNode{T}(dim, nothing)
RootNode{T}(input::Array{T}) where T = RootNode{T}(size(input)[end], input) 
RootNode(input::Array) = RootNode{eltype(input)}(input)

"Check the type of a CG node"
gettype(::NT) where NT <: CompNode{T} where T <: Real = T

"Set the input of a root node."
function setinput!(node::RootNode{T}, input::Array{T}) where T <: Real
    node.input = input
end

"Get the output of a node"
output!(node::RootNode{T}) where T <: Real = node.input

""" A generic tensor-valued node in a computational graph. """
mutable struct Node{T<:Real} <: CompNode{T}
    dim::Int
    inpdim::Int
    input::CompNode{T}
    level::Int
    activation::Symbol
    weights::Variable{T, 2}
    bias::Variable{T,1}
    
    _cachedinput::Array{T}
    #not a jacobian bcs I'm assuming layerwise activation
    _cachedvectorjac::Array{T}
    #index of the Node within given topological layer
    _layerindex::Int    

    function Node{T}(dim::Int, input::CompNode{T}; activation::Symbol = :identity, init::Symbol = :randn) where T<: Real
        inpdim = input.dim
        level = input.level + 1
        initializer = getinitializer(init)
        weights = Variable(initializer(T, dim, inpdim))
        bias = Variable(initializer(T, dim))
        _cachedinput = zeros(T, inpdim)
        _cachedvectorjac = zeros(T, dim)
        _layerindex = 0
        
        return new(dim, inpdim, input, level, activation, weights, bias, _cachedinput, _cachedvectorjac, _layerindex)
    end
end

"Update a node's weight tensor."
function setweights!(node::Node{T}, data::Array{T, 2}) where T <: Real
    set!(node.weights, data)
end

"Update a node's bias tensor"
function setbias!(node::Node{T}, data::Array{T, 1}) where T <: Real
    set!(node.bias, data)
end

function Node(dim::Int, input::CompNode{T}; activation::Symbol=:identity, init::Symbol=:randn) where T<:Real 
    return Node{T}(dim, input; activation=activation, init=init)
end

function setlayerindex!(node::CompNode, index::Int)
    node._layerindex = index
end

" Apply linear transformation to a one-dimensional input vector "
function applylinear(weights::Array{T, 2}, bias::Array{T, 1}, input::Array{T, 1}) where T<:Real
    return weights * input + bias
end

" This is for batched inputs. Batch dimension comes first"
function applylinear(weights::Array{T, 2}, bias::Array{T, 1}, input::Array{T, 2}) where T<: Real
    b = reshape(bias, (1, length(bias)))
    return broadcast(+, input * transpose(weights), b)
end

function applylinear(weights::Variable, bias::Variable, input::Array) 
    return applylinear(weights.data, bias.data, input)
end

"Apply weight and bias to input tensor, and cache the result at the node."
function computelinear!(node::Node{T}, input::Array{T}) where T <: Real
    s = applylinear(node.weights, node.bias, input)
    node._cachedinput = input
    return s
end

"Given symbol representing activation function, return symbol representing gradient.
Currently base functions only.
Returns: expression :(fprime(x))"
function getgradient(activation::Symbol)::Function
    return _ACTIVATION_GRADIENTS[activation]
end

function getactivation(activation::Symbol)::Function
    return _ACTIVATIONS[activation]
end

""" Computes the tensor which results from the action of the node upon the given input array"""
function apply!(node::CompNode{T}, input::Array{T}) where {T<:Real}
    s = computelinear!(node, input)
    gradfn = getgradient(node.activation)
    actfn = getactivation(node.activation)
    node._cachedvectorjac = gradfn.(s)
    return actfn.(s)
end

"""Compute the node's forward pass: output tensor from its input node, cache input and jacobian, and
return output."""
function forward!(node::Node{T}) where T <: Real
    input = output!(node.input)
    return apply!(node, input)
end

function output!(node::Node{T}) where T <: Real
    return forward!(node)
end

""" Holds a computational graph """
mutable struct Graph{T}
    rootnodes::Array{RootNode{T}, 1}
    nodes::Array{Array{CompNode{T}, 1},1}
    function Graph{T}() where T <: Real
        rootnodes = Array{RootNode{T},1}()
        nodes = Array{Array{CompNode{T}}, 1}()
        return new(rootnodes, nodes)
    end
end

numroots(graph::Graph) = length(graph.rootnodes)
"Number of layers in the topological sort of the graph, not including the inputs."
numlayers(graph::Graph) = length(graph.nodes)

"Add an empty layer to the graph"
function addlayer!(graph::Graph{T}) where T <: Real
    push!(graph.nodes, Array{CompNode{T}, 1}() )
end


"Add a node at the specified level in the topological sort.
    `level`: topological-sort level at which to add the node. >= 0 (0 means input layer)"
function addnode!(graph::Graph{T}, node::CompNode{T}, level::Int) where T <: Real
    (level < 0 || level > numlayers(graph)) && throw(ArgumentError("$level is not valid for graph with $(numlayers(graph)) layers."))
    level == 0 && (isa(node, RootNode{T}) || throw(ArgumentError("Only root nodes can be added to level 0")))

    node._layerindex != 0 && throw(ArgumentError("This node has already been added to a graph!"))
    if level == 0
        nodelist = graph.rootnodes
    else
        nodelist = graph.nodes[level]
    end
    layerindex = length(nodelist) + 1
    setlayerindex!(node, layerindex)
    push!(nodelist, node)
end


function addnode!(graph::Graph{T}, node::CompNode{T}) where T <: Real
    toplevel = numlayers(graph)
    if node.level > toplevel + 1
        throw(ArgumentError("Node level is $(node.level) but graph has only $toplevel"))
    end
    if node.level > toplevel
        addlayer!(graph)
    end
    addnode!(graph, node, node.level)
end

"By default only the highest level of the graph is deemed output."
function numoutputs(graph::Graph) 
    numlayers(graph) == 0 && return 0
    return length(graph.nodes[end])
end

"Performs a forward pass of the graph. Starting from the lowest graph level and moving upwards, 
node outputs are computed, and inputs and jacobians are cached."
function forward!(graph::Graph{T}) where T<: Real
    any(map(rn -> isnothing(output!(rn)), graph.rootnodes)) && throw(ErrorException("Graph roots have not been initialized!"))
    prev_outputs = [output!(rn) for rn in graph.rootnodes]
    cur_outputs = Array{Array{T}, 1}()
    for i in 1:numlayers(graph)
        cur_outputs = Array{Array{T},1}()
        for j in 1:length(graph.nodes[i])
            curnode = graph.nodes[i][j]
            inp_index = curnode.input._layerindex
            output = apply!(curnode, prev_outputs[inp_index])
            push!(cur_outputs, output)
        end
        prev_outputs = cur_outputs
        
    end
    return cur_outputs
end
