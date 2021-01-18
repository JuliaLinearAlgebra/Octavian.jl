@inline cld_fast(x, y) = cld(x, y)
@inline function cld_fast(x::I, y) where {I <: Integer}
    d = div_fast(x, y)
    (d + (d * unsigned(y) != unsigned(x))) % I
end
cld_fast(::StaticInt{N}, ::StaticInt{M}) where {N,M}= (StaticInt{N}() + StaticInt{M}() + One()) ÷ StaticInt{M}()
@inline function divrem_fast(x::I, y) where {I <: Integer}
    ux = unsigned(x); uy = unsigned(y)
    d = Base.udiv_int(ux, uy)
    r = ux - d * uy
    d % I, r % I
end
@inline divrem_fast(::StaticInt{x}, y::I) where {x, I <: Integer} = divrem_fast(x % I, y)
@inline div_fast(x::I, y::Integer) where {I <: Integer} = Base.udiv_int(unsigned(x), unsigned(y)) % I
@inline div_fast(::StaticInt{x}, y::I) where {x, I <: Integer} = Base.udiv_int(unsigned(x), unsigned(y)) % I
divrem_fast(::StaticInt{N}, ::StaticInt{M}) where {N,M}= divrem(StaticInt{N}(), StaticInt{M}())
div_fast(::StaticInt{N}, ::StaticInt{M}) where {N,M}= StaticInt{N}() ÷ StaticInt{M}()
@generated function div_fast(x::I, ::StaticInt{M}) where {I<:Integer,M}
    if VectorizationBase.ispow2(M)
        lm = VectorizationBase.intlog2(M)
        Expr(:block, Expr(:meta,:inline), :(x >>> $lm))
    else
        Expr(:block, Expr(:meta,:inline), :(div_fast(x, $(I(M)))))
    end
end

