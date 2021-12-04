n_values  = [1, 10, 20, 50, 100, 150, 200]
k_values  = [10, 20, 50, 100, 150, 200]
m_values  = [10, 20, 50, 100, 150, 200]



typ = get(ENV, "JULIA_TEST_ELTYPE", "ALL")
types = if typ == "Float64"
  DataType[Float64]
elseif typ == "Float32"
  DataType[Float32]
elseif typ == "Int64"
  DataType[Int64]
elseif typ == "Int32"
  DataType[Int32]
else
  DataType[Float64, Float32, Int64, Int32]
end
testset_name_suffix = "(coverage)"
for T ∈ types
  @time test_complex(T, m_values, k_values, n_values, testset_name_suffix)
  @time test_real(T, m_values, k_values, n_values, testset_name_suffix)
end

