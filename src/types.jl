struct BCache{T<:Union{UInt,Nothing}}
    p::Ptr{Cvoid}
    i::T
end


