```@meta
CurrentModule = Octavian
```

# Getting Started

## Multi-threaded matrix multiplication: `matmul!` and `matmul`

Remember to start Julia with multiple threads with e.g. one of the following:
- `julia -t auto`
- `julia -t 4`
- Set the `JULIA_NUM_THREADS` environment variable to `4` **before** starting Julia

```@repl
using Octavian

A = [1 2 3; 4 5 6]

B = [7 8 9 10; 11 12 13 14; 15 16 17 18]

C = Matrix{Int}(undef, 2, 4)

Octavian.matmul!(C, A, B) # (multi-threaded) multiply A×B and store the result in C (overwriting the contents of C)

C == A * B
```

```@repl
using Octavian

A = [1 2 3; 4 5 6]

B = [7 8 9 10; 11 12 13 14; 15 16 17 18]

C = Octavian.matmul(A, B) # (multi-threaded) multiply A×B and return the result

C == A * B
```
