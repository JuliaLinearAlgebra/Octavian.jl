n_values  = [1, 10, 20, 50, 100, 150, 200]
k_values  = [10, 20, 50, 100, 150, 200]
m_values  = [10, 20, 50, 100, 150, 200]

function matmul_pack_ab!(C, A, B)
  M, N = size(C); K = size(B,1)
  zc, za, zb = Octavian.zstridedpointer.((C,A,B))
  nspawn = min(Threads.nthreads(), Octavian.num_cores())
  GC.@preserve C A B begin
    if nspawn > 1
      threads, torelease = Octavian.PolyesterWeave.__request_threads((nspawn-1)%UInt32, Octavian.PolyesterWeave.worker_pointer())
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

testset_name_suffix = "(coverage)"

include("_matmul.jl")
