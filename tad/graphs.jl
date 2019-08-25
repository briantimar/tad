"""Struct representing a node (operation) in a computational graph"""

_DEFAULT_FLOAT = Float32

abstract type GraphElement end
abstract type AbstractNode <: GraphElement end
abstract type CompNode{T<:Real} <: AbstractNode end


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
    # inputs:: Tuple{Vararg{Union{Node{T}, RootNode{T}}}}
    inputs:: Array{CompNode{T}, 1}
    activation::Symbol

    function Node{T}(dim::Int, inputs::Array{CompNode{T},1}, activation::Symbol = :identity) where T<: Real
        inpdim = sum(input.dim for input in inputs)
        return new(dim, inpdim, inputs, activation)
    end
end

function Node{T}(dim::Int, inputs::Array{RootNode{T}, 1}, activation::Symbol) where T <: Real 
    return Node{T}(dim, convert(Array{CompNode{T},1}, inputs), activation)
end

function Node{T}(dim::Int, inputs::Array{Node{T}, 1}, activation::Symbol) where T <: Real 
    return Node{T}(dim, convert(Array{CompNode{T},1}, inputs), activation)
end