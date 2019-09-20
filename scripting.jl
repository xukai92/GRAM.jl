function get_args(
    dataset, 
    model_name; 
    override::Dict{Symbol,<:Any}=Dict{Symbol,Any}(),
    suffix::String=""
)
    @assert dataset in [
        "gaussian", 
        "ring", 
        "mnist"
    ]
    
    @assert model_name in [
        "mmdnet", 
        "rmmmdnet"
    ]

    args_dict = Dict{Symbol,Any}(
        :dataset => dataset,
        :model_name => model_name,
        :batch_size => 500,
        :batch_size_gen => 500,
        :lr => 1f-3
    )
    if model_name == "mmdnet"
        args_dict[:opt] = "rmsprop"
    elseif model_name == "rmmmdnet"
        args_dict[:opt] = "adam"
    end

    if dataset == "gaussian"
        args_dict[:n_epochs] = 200
        args_dict[:D_h] = "50,50"
        args_dict[:D_z] = 5
        args_dict[:σ] = "tanh"
        args_dict[:σ_last] = "identity"
        args_dict[:σs] = "1,2"
        if model_name == "rmmmdnet"
            args_dict[:D_fx] = 2
        end
    end
    
    if dataset == "ring"
        args_dict[:n_epochs] = 400
        args_dict[:D_h] = "200,200,200"
        args_dict[:D_z] = 10
        args_dict[:σ] = "tanh"
        args_dict[:σ_last] = "identity"
        args_dict[:σs] = "1,2,4"
        if model_name == "rmmmdnet"
            args_dict[:D_fx] = 1
        end
    end
    
    if dataset == "mnist"
        args_dict[:n_epochs] = 10
        args_dict[:D_h] = "200,400,800"
        args_dict[:D_z] = 100
        args_dict[:σ] = "relu"
        args_dict[:σ_last] = "sigmoid"
        args_dict[:σs] = "1,5,10"
        if model_name == "rmmmdnet"
            args_dict[:D_fx] = 100
        end
    end
    
    # Oeverride
    for k in keys(override)
        args_dict[k] = override[k]
    end
    
    # Show arguments
    @info "On dataset $dataset with args" args_dict...

    # Generate experiment name from dict
    exclude = [
        :dataset,
        :model_name,
        :n_epochs,
        :batch_size, 
        :batch_size_gen, 
        :lr, 
        :σ_last,
    ]
    exp_name = flatten_dict(args_dict; exclude=exclude)
    if !isnothing(suffix) && suffix != ""
        exp_name *= "-$suffix"
    end

    # Parse "1,2,3" => [1,2,3]
    args_dict[:σs] = if args_dict[:σs] == "median" 
        []
    else
        map(
            x -> parse(Float32, x), 
            split(args_dict[:σs], ",")
        )
    end
    args_dict[:D_h] = map(
        x -> parse(Int, x), 
        split(args_dict[:D_h], ",")
    )

    # Convert dict to named tuple
    args = dict2namedtuple(args_dict)
    
    return args, exp_name
end

