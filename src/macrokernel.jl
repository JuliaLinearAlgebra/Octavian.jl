
"""
The macrokernel. It iterates over our tiles, and applies the microkernel.
"""
function macrokernel!(C, A, B, α, β)
    LoopVectorization.@avx for n ∈ axes(C,2), m ∈ axes(C,1)
        Cmn = zero(eltype(C))
        for k ∈ axes(B,1)
            Cmn += A[m,k] * B[k,n]
        end
        # C[m,n] may be `NaN`, but if `β == 0` we still want to zero it
        C[m,n] = (α * Cmn) + (β == zero(β) ? zero(β) : β * C[m,n])
    end
end

