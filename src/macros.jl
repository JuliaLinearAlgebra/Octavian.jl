# Some of the code in this file is taken from Gaius.jl (license: MIT)
# https://github.com/MasonProtter/Gaius.jl

# Note this does not support changing the number of threads at runtime

macro _spawn(ex)
    if Threads.nthreads() > 1
        return esc(Expr(:macrocall, Expr(:(.), :Threads, QuoteNode(Symbol("@spawn"))), __source__, ex))
    else
        return esc(ex)
    end
end

macro _sync(ex)
    if Threads.nthreads() > 1
        return esc(Expr(:macrocall, Symbol("@sync"), __source__, ex))
    else
        return esc(ex)
    end
end
