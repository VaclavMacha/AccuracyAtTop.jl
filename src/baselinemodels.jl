mutable struct BaseLine <: BaseLineModel
    classifier

    BaseLine(classifier) = new(deepcopy(classifier))
end


show(io::IO, model::BaseLine) = print(io, "BaseLine")
