using Turing

include("config.jl")

for model in modellist
  include("$model/model.jl")
end
