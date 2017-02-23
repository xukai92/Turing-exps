using Turing

include("modellist.jl")

for model in modellist
  include("$model/model.jl")
end
