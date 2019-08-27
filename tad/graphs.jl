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

    function RootNode{T}(dim::Int, input::Union{Array{T}, Nothing}) where T <: Real 
        isnothing(input)  || (! checkvalidnodedim(input, dim) && throw(ArgumentError("Input has invalid shape $(size(input)) for node dim $dim")) );
        return new(dim, input)
    end

end

RootNode(dim, input) = RootNode{_DEFAULT_FLOAT}(dim, input)
RootNode{T}(dim) where T = RootNode{T}(dim, nothing)

"Check the type of a CG node"
gettype(::NT) where NT <: CompNode{T} where T <: Real = T

"Set the input of a root node."
function setinput!(node::RootNode{T}, input::Array{T}) where T <: Real
    node.input = input
end

"Get the output of a node"
output(node::RootNode{T}) where T <: Real = node.input

""" A generic tensor-valued node in a computational graph. """
mutable struct Node{T<:Real} <: CompNode{T}
    dim::Int
    inpdim::Int
    input::CompNode{T}
    activation::Symbol
    weights::Variable{T, 2}
    bias::Variable{T,1}
    
    _cachedinput::Array{T, 1}
    #not a jacobian bcs I'm assuming layerwise activation
    _cachedvectorjac::Array{T, 1}

    function Node{T}(dim::Int, input::CompNode{T}, activation::Symbol = :identity) where T<: Real
        inpdim = input.dim
        weights = Variable(randn(T, dim, inpdim))
        bias = Variable(randn(T, dim))
        _cachedinput = zeros(T, inpdim)
        _cachedvectorjac = zeros(T, dim)
        return new(dim, inpdim, input, activation, weights, bias, _cachedinput, _cachedvectorjac)
    end
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
function computelinear!(node::Node{T}, input::Array{T, 1}) where T <: Real
    s = applylinear(node.weights, node.bias, input)
    node._cachedinput .= input
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
