mutable struct MemoryBuffer{L,T} <: DenseVector{T}
    data::NTuple{L,T}
    @inline function MemoryBuffer{L,T}(::UndefInitializer) where {L,T}
        @assert isbitstype(T) "Memory buffers must point to bits types, but `isbitstype($T) == false`."
        new{L,T}()
    end
end

struct StaticFloat{N} <: AbstractFloat
    StaticFloat{N}() where {N} = new{N::Float64}()
end



