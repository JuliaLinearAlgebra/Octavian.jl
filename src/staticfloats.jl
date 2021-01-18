Base.convert(::Type{T}, ::StaticFloat{N}) where {N,T<:AbstractFloat} = T(N)
Base.promote_rule(::Type{StaticFloat{N}}, ::Type{T}) where {N,T} = promote_type(T, Float64)

@generated Base.:+(::StaticFloat{M}, ::StaticFloat{N}) where {M,N} = Expr(:call, Expr(:curly, :StaticFloat, M+N))
@generated Base.:+(::StaticFloat{N}, ::StaticInt{M}) where {M,N} = Expr(:call, Expr(:curly, :StaticFloat, M+N))
@generated Base.:+(::StaticInt{M}, ::StaticFloat{N}) where {M,N} = Expr(:call, Expr(:curly, :StaticFloat, M+N))

@generated Base.:+(::StaticFloat{N}, ::StaticInt{0}) where {N} = Expr(:call, Expr(:curly, :StaticFloat, N))
@generated Base.:+(::StaticInt{0}, ::StaticFloat{N}) where {N} = Expr(:call, Expr(:curly, :StaticFloat, N))

@generated Base.:-(::StaticFloat{M}, ::StaticFloat{N}) where {M,N} = Expr(:call, Expr(:curly, :StaticFloat, M - N))
@generated Base.:-(::StaticFloat{N}, ::StaticInt{M}) where {M,N} = Expr(:call, Expr(:curly, :StaticFloat, N - M))
@generated Base.:-(::StaticInt{M}, ::StaticFloat{N}) where {M,N} = Expr(:call, Expr(:curly, :StaticFloat, M - N))

@generated Base.:-(::StaticFloat{N}, ::StaticInt{0}) where {N} = Expr(:call, Expr(:curly, :StaticFloat, N))
@generated Base.:-(::StaticInt{0}, ::StaticFloat{N}) where {N} = Expr(:call, Expr(:curly, :StaticFloat, -N))


@generated Base.:*(::StaticFloat{M}, ::StaticFloat{N}) where {M,N} = Expr(:call, Expr(:curly, :StaticFloat, M*N))
@generated Base.:*(::StaticFloat{N}, ::StaticInt{M}) where {M,N} = Expr(:call, Expr(:curly, :StaticFloat, M*N))
@generated Base.:*(::StaticInt{M}, ::StaticFloat{N}) where {M,N} = Expr(:call, Expr(:curly, :StaticFloat, M*N))

@generated Base.:*(::StaticFloat{N}, ::StaticInt{0}) where {N} = Expr(:call, Expr(:curly, :StaticInt, 0))
@generated Base.:*(::StaticInt{0}, ::StaticFloat{N}) where {N} = Expr(:call, Expr(:curly, :StaticInt, 0))
@generated Base.:*(::StaticFloat{N}, ::StaticInt{1}) where {N} = Expr(:call, Expr(:curly, :StaticFloat, N))
@generated Base.:*(::StaticInt{1}, ::StaticFloat{N}) where {N} = Expr(:call, Expr(:curly, :StaticFloat, N))

@generated Base.:/(::StaticFloat{M}, ::StaticFloat{N}) where {M,N} = Expr(:call, Expr(:curly, :StaticFloat, M/N))
@generated Base.:/(::StaticFloat{N}, ::StaticInt{M}) where {M,N} = Expr(:call, Expr(:curly, :StaticFloat, N/M))
@generated Base.:/(::StaticInt{M}, ::StaticFloat{N}) where {M,N} = Expr(:call, Expr(:curly, :StaticFloat, M/N))

@generated Base.sqrt(::StaticInt{M}) where {M} = Expr(:call, Expr(:curly, :StaticFloat, sqrt(M)))
@generated Base.sqrt(::StaticFloat{M}) where {M} = Expr(:call, Expr(:curly, :StaticFloat, sqrt(M)))

@generated Base.round(::StaticFloat{M}) where {M} = Expr(:call, Expr(:curly, :StaticFloat, round(M)))
@generated roundtostaticint(::StaticFloat{M}) where {M} = Expr(:call, Expr(:curly, :StaticInt, round(Int, M)))
roundtostaticint(x::AbstractFloat) = round(Int, x)
@generated floortostaticint(::StaticFloat{M}) where {M} = Expr(:call, Expr(:curly, :StaticInt, floor(Int, M)))
floortostaticint(x::AbstractFloat) = Base.fptosi(Int, x)
@generated Base.inv(::StaticFloat{N}) where {N} = Expr(:call, Expr(:curly, :StaticFloat, inv(N)))

