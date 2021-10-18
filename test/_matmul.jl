# The following variables need to be defined before `include`-ing this file:
# `testset_name_suffix`
# `n_values`
# `k_values`
# `m_values`
function test_complex(::Type{TE}, m_values, k_values, n_values, testset_name_suffix) where {TE}
  T = Complex{TE}
  @time @testset "Matrix Multiply $T $(testset_name_suffix)" begin
    for n ∈ n_values
      for k ∈ k_values
        for m ∈ m_values
          A = rand(T, m, k);
          B = rand(T, k, n);
          b = rand(T, k);

          Are = real.(A);
          Bre = real.(B);
          bre = real.(b);

          A′ = permutedims(A)'
          B′ = permutedims(B)'
          AB = A * B;
          A′B = A′*B;
          AB′ = A*B′;
          A′B′= A′*B′;
          Ab = A*b;
          A′b = A′*b;

          AreB = Are*B;
          ABre = A*Bre;
          Areb = Are*b;
          Abre = A*bre;

          @info "" T n k m
          @test @time(Octavian.matmul(A, B)) ≈ AB
          @test @time(Octavian.matmul(A, Bre)) ≈ ABre
          @test @time(Octavian.matmul(Are, B)) ≈ AreB
          @test @time(Octavian.matmul(A′, B)) ≈ A′B
          @test @time(Octavian.matmul(A, B′)) ≈ AB′
          @test @time(Octavian.matmul(A′, B′)) ≈ A′B′

          @test @time(Octavian.matmul(A, b)) ≈ Ab
          @test transpose(@time(Octavian.matmul(transpose(b), transpose(A)))) ≈ Ab
          @test @time(Octavian.matmul(A, bre)) ≈ Abre
          @test @time(Octavian.matmul(Are, b)) ≈ Areb
          @test @time(Octavian.matmul(A′, b)) ≈ A′b
          @test transpose(@time(Octavian.matmul(transpose(b), transpose(A′)))) ≈ A′b

          @test @time(Octavian.matmul_serial(A, B)) ≈ AB
          @test @time(Octavian.matmul_serial(A, Bre)) ≈ ABre
          @test @time(Octavian.matmul_serial(Are, B)) ≈ AreB
          @test @time(Octavian.matmul_serial(A′, B)) ≈ A′B
          @test @time(Octavian.matmul_serial(A, B′)) ≈ AB′
          @test @time(Octavian.matmul_serial(A′, B′)) ≈ A′B′

          @test @time(Octavian.matmul_serial(A, b)) ≈ Ab
          @test transpose(@time(Octavian.matmul_serial(transpose(b), transpose(A)))) ≈ Ab
          @test @time(Octavian.matmul_serial(A, bre)) ≈ Abre
          @test @time(Octavian.matmul_serial(Are, b)) ≈ Areb
          @test @time(Octavian.matmul_serial(A′, b)) ≈ A′b
          @test transpose(@time(Octavian.matmul_serial(transpose(b), transpose(A′)))) ≈ A′b

          C = Matrix{T}(undef, n, m)'
          @test @time(Octavian.matmul!(C, A, B)) ≈ AB

          C1 = rand(T, m, n)
          C2 = copy(C1)
          α, β = T(1 - 2im), T(3 + 4im)
          @test @time(Octavian.matmul!(C1, A, B, α, β)) ≈ Octavian.matmul!(C2, A, B, α, β)
        end
      end
    end
  end
end

function matmul_pack_ab!(C, A, B)
  M, N = size(C); K = size(B,1)
  zc, za, zb = Octavian.zstridedpointer.((C,A,B))
  nspawn = min(Threads.nthreads(), Octavian.num_cores())
  GC.@preserve C A B begin
    if nspawn > 1
      threads, torelease = Octavian.PolyesterWeave.__request_threads((nspawn-1)%UInt32, Octavian.PolyesterWeave.worker_pointer(), nothing)
      @assert threads.i < Threads.nthreads()
      Octavian.matmul_pack_A_and_B!(
        zc, za, zb, Octavian.StaticInt{1}(), Octavian.StaticInt{0}(), M, K, N, threads,
        Octavian.W₁Default(), Octavian.W₂Default(), Octavian.R₁Default(), Octavian.R₂Default()
      )
      Octavian.PolyesterWeave.free_threads!(torelease)
    else
      Octavian.matmul_st_pack_A_and_B!(
        zc, za, zb, Octavian.StaticInt{1}(), Octavian.StaticInt{0}(), M, K, N,
        Octavian.W₁Default(), Octavian.W₂Default(), Octavian.R₁Default(), Octavian.R₂Default(), 0
      )
    end
  end
  C
end

function test_real(::Type{T}, m_values, k_values, n_values, testset_name_suffix) where {T}
  @time @testset "Matrix Multiply $T $(testset_name_suffix)" begin
    for n ∈ n_values
      for k ∈ k_values
        for m ∈ m_values
          A = rand(T, m, k)
          B = rand(T, k, n)
          b = rand(T, k)
          A′ = permutedims(A)'
          B′ = permutedims(B)'
          AB = A * B;
          Ab = A * b;
          @info "" T n k m
          @test @time(Octavian.matmul(A, B)) ≈ AB
          @test @time(Octavian.matmul(A′, B)) ≈ AB
          @test @time(Octavian.matmul(A, B′)) ≈ AB
          @test @time(Octavian.matmul(A′, B′)) ≈ AB
          @test @time(Octavian.matmul_serial(A, B)) ≈ AB
          @test @time(Octavian.matmul_serial(A′, B)) ≈ AB
          @test @time(Octavian.matmul_serial(A, B′)) ≈ AB
          @test @time(Octavian.matmul_serial(A′, B′)) ≈ AB
          @test @time(Octavian.matmul(A, b)) ≈ Ab
          @test @time(Octavian.matmul(A′, b)) ≈ Ab
          @test @time(Octavian.matmul(b', A'))' ≈ Ab
          @test @time(Octavian.matmul(b', A′'))' ≈ Ab
          @test @time(Octavian.matmul_serial(A, b)) ≈ Ab
          @test @time(Octavian.matmul_serial(A′, b)) ≈ Ab
          @test @time(Octavian.matmul_serial(b', A'))' ≈ Ab
          @test @time(Octavian.matmul_serial(b', A′'))' ≈ Ab
        end
      end
    end
    m = k = n = max(8Octavian.OCTAVIAN_NUM_TASKS[], 400)
    A = rand(T, m, k);
    B = rand(T, k, n);
    A′ = permutedims(A)';
    B′ = permutedims(B)';
    AB = A * B;
    @test matmul_pack_ab!(similar(AB), A, B) ≈ AB
    @test matmul_pack_ab!(similar(AB), A, B′) ≈ AB
    @test matmul_pack_ab!(similar(AB), A′, B) ≈ AB
    @test matmul_pack_ab!(similar(AB), A′, B′) ≈ AB
  end
end

@time @testset "zero-sized-matrices" begin
    @test Octavian.matmul_serial(randn(0,0), randn(0,0)) == zeros(0, 0)
    @test Octavian.matmul_serial(randn(2,3), randn(3,0)) == zeros(2, 0)
    @test Octavian.matmul_serial(randn(2,0), randn(0,2)) == zeros(2, 2)
    @test Octavian.matmul_serial!(ones(2,2),randn(2,0), randn(0,2), 1.0, 2.0) == ones(2, 2) .* 2
    @test Octavian.matmul(randn(0,0), randn(0,0)) == zeros(0, 0)
    @test Octavian.matmul(randn(2,3), randn(3,0)) == zeros(2, 0)
    @test Octavian.matmul(randn(2,0), randn(0,2)) == zeros(2, 2)
    @test Octavian.matmul!(ones(2,2),randn(2,0), randn(0,2), 1.0, 2.0) == ones(2, 2) .* 2
end

