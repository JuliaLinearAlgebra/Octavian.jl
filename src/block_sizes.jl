"""
  block_sizes(::Type{T}) -> (Mc, Kc, Nc)

Returns the dimensions of our macrokernel, which iterates over the microkernel. That is, in
calculating `C = A * B`, our macrokernel will be called on `Mc × Kc` blocks of `A`, multiplying
 them with `Kc × Nc` blocks of `B`, to update `Mc × Nc` blocks of `C`.

We want these blocks to fit nicely in the cache. There is a lot of room for improvement here, but this
initial implementation should work reasonably well.


The constants `LoopVectorization.mᵣ` and `LoopVectorization.nᵣ` are the factors by which LoopVectorization
wants to unroll the rows and columns of the microkernel, respectively. `LoopVectorization` defines these
constants by running it's analysis on a gemm kernel; they're there for convenience/to make it easier to
implement matrix-multiply.
It also wants to vectorize the rows by `W = VectorizationBase.pick_vector_width_val(T)`.
Thus, the microkernel's dimensions are `(W * mᵣ) × nᵣ`; that is, the microkernel updates a `(W * mᵣ) × nᵣ`
block of `C`.

Because the macrokernel iterates over tiles and repeatedly applies the microkernel, we would prefer the
macrokernel's dimensions to be an integer multiple of the microkernel's.
That is, we want `Mc` to be an integer multiple of `W * mᵣ` and `Nc` to be an integer multiple of `nᵣ`.

Additionally, we want our blocks of `A` to fit in the core-local L2 cache.
Empirically, I found that when using `Float64` arrays, 72 rows works well on Haswell (where `W * mᵣ = 8`)
and 96 works well for Cascadelake (where `W * mᵣ = 24`).
So I kind of heuristically multiply `W * mᵣ` by `4` given 32 vector register (as in Cascadelake), which
would yield `96`, and multiply by `9` otherwise, which would give `72` on Haswell.
Ideally, we'd have a better means of picking.
I suspect relatively small numbers work well because I'm currently using a column-major memory layout for
the internal packing arrays. A column-major memory layout means that if our macro-kernel had a lot of rows,
moving across columns would involve reading memory far apart, moving across memory pages more rapidly,
hitting the TLB harder. This is why libraries like OpenBLAS and BLIS don't use a column-major layout, but
reshape into a 3-d array, e.g. `A` will be reshaped into a `Mᵣ × Kc × (Mc ÷ Mᵣ)` array (also sometimes
referred to as a tile-major matrix), so that all memory reads happen in consecutive memory locations.


Now that we have `Mc`, we use it and the `L2` cache size to calculate `Kc`, but shave off a percent to
leave room in the cache for some other things.

We want out blocks of `B` to fir in the `L3` cache, so we can use the `L3` cache-size and `Kc` to
similarly calculate `Nc`, with the additional note that we also divide and multiply by `nᵣ` to ensure
that `Nc` is an integer multiple of `nᵣ`.
"""
function block_sizes(::Type{T}) where {T}
    _L1, _L2, __L3, _L4 = VectorizationBase.CACHE_SIZE
    L1c, L2c, L3c, L4c = VectorizationBase.CACHE_COUNT
    # TODO: something better than treating it as 4 MiB if there is no L3 cache
    #       one possibility is to focus on the L1 and L2 caches instead of the L2 and L3.
    _L3 = something(__L3, 4194304)
    @assert L1c == L2c == VectorizationBase.NUM_CORES

    st = VectorizationBase.static_sizeof(T)

    L2 = (StaticInt{_L2}() - StaticInt{_L1}()) ÷ st
    if 2_L2 * L2c > _L3 * L3c
        L3 = StaticInt{_L3}() ÷ st
    else
        L3 = (StaticInt{_L3}() - StaticInt{_L2}()) ÷ st
    end

    W = VectorizationBase.pick_vector_width_val(T)
    Mr = StaticInt{LoopVectorization.mᵣ}()
    Nr = StaticInt{LoopVectorization.nᵣ}()

    Mc = (VectorizationBase.REGISTER_COUNT == 32 ? StaticInt{4}() : StaticInt{9}()) * Mr * W
    Kc = (StaticInt{5}() * L2) ÷ (StaticInt{7}() * Mc)
    Nc = ((StaticInt{5}() * L3) ÷ (StaticInt{7}() * Kc * Nr)) * Nr

    Mc, Kc, Nc
end
