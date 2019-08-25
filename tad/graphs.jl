"""Struct representing a node (operation) in a computational graph"""

_DEFAULT_FLOAT = Float32

abstract type GraphElement end
abstract type AbstractNode <: GraphElement end
abstract type CompNode{T<:Real} <: AbstractNode end


"Holds inputs to graph"
mutable struct RootNode{T<:Real}
    dim::Int
end

"Check the type of a CG node"
gettype(::RootNode{T}) where T <: Real = T

RootNode(dim) = RootNode{_DEFAULT_FLOAT}(dim)
