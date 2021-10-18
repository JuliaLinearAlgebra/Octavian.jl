n_values  = [200, 300, 400]
k_values  = [200, 300, 400]
m_values  = [200, 300, 400]

testset_name_suffix = "(main)"

for T ∈ (Float64,Float32,Int64,Int32)
  @time test_complex(T, m_values, k_values, n_values, testset_name_suffix)
  @time test_real(T, m_values, k_values, n_values, testset_name_suffix)
end

