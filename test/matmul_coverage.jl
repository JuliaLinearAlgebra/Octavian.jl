n_values  = [10, 20, 50]
k_values  = [10, 20, 50]
m_values  = [10, 20, 50]

additional_n_k_m_values = [
    (2000, 2000, 2000),
]

testset_name_suffix = "(coverage)"

include("_matmul.jl")
