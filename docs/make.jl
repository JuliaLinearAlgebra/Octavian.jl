using Augustus
using Documenter

makedocs(;
    modules=[Augustus],
    authors="Mason Protter, Chris Elrod, Dilum Aluthge, and contributors",
    repo="https://github.com/JuliaLinearAlgebra/Augustus.jl/blob/{commit}{path}#L{line}",
    sitename="Augustus.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://JuliaLinearAlgebra.github.io/Augustus.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
        "API" => "api.md",
    ],
    strict=true,
)

deploydocs(;
    repo="github.com/JuliaLinearAlgebra/Augustus.jl",
)
