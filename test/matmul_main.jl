n_values  = [200, 300, 400]
k_values  = [200, 300, 400]
m_values  = [200, 300, 400]

testset_name_suffix = "(main)"

for T ∈ (Float64,Float32,Int64,Int32)
  @time test_complex(T, m_values, k_values, n_values, testset_name_suffix)
  @time test_real(T, m_values, k_values, n_values, testset_name_suffix)
end

A = rand(2,2); B = rand(2,2); AB = A*B; C = fill(NaN, 2, 2);
@test Octavian.matmul!(C, A, B, true, false) ≈ AB
@test Octavian.matmul!(C, A, B, true, true) ≈ 2AB

