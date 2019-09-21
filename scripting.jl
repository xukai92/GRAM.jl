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
        :seed => 1234,
        :dataset => dataset,
        :model_name => model_name,
        :batch_size => 200,
        :batch_size_gen => 200,
    )
    if model_name == "mmdnet"
        args_dict[:opt] = "rmsprop"
    elseif model_name == "rmmmdnet"
        args_dict[:opt] = "adam_akash"
    end

    if dataset == "gaussian"
        args_dict[:n_epochs] = 100
        args_dict[:lr] = 1f-3
        args_dict[:base] = "uniform"
        args_dict[:D_z] = 10
        args_dict[:Dg_h] = "50,50"
        args_dict[:σ] = "tanh"
        args_dict[:σ_last] = "identity"
        args_dict[:σs] = "1,2"
        if model_name == "rmmmdnet"
            args_dict[:Df_h] = "50,25"
            args_dict[:D_fx] = 5
        end
    end
    
    if dataset == "ring"
        args_dict[:n_epochs] = 200
        args_dict[:lr] = 1f-3
        args_dict[:base] = "gaussian"
        args_dict[:D_z] = 256
        args_dict[:Dg_h] = "128"
        args_dict[:σ] = "relu"
        args_dict[:σ_last] = "identity"
        args_dict[:σs] = "1"
        if model_name == "rmmmdnet"
            args_dict[:Df_h] = "128"
            args_dict[:D_fx] = 2
        end
    end
    
    if dataset == "mnist"
        args_dict[:n_epochs] = 10
        args_dict[:base] = "uniform"
        args_dict[:D_z] = 100
        args_dict[:Dg_h] = "200,400,800"
        args_dict[:σ] = "relu"
        args_dict[:σ_last] = "sigmoid"
        if model_name == "mmdnet"
            args_dict[:lr] = 1f-3
            args_dict[:σs] = "1,5,10"
        elseif model_name == "rmmmdnet"
            args_dict[:lr] = 1f-4
            args_dict[:σs] = "1,2,4,8,16"
            args_dict[:Df_h] = "400,200,100"
            args_dict[:D_fx] = 10
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
        :seed,
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

    # Add exp_name to args_dict
    args_dict[:exp_name] = exp_name

    # Parse "1,2,3" => [1,2,3]
    parse_csv(T, l) = map(x -> parse(T, x), split(l, ","))

    args_dict[:σs] = if args_dict[:σs] == "median" 
        []
    else
        parse_csv(Float32, args_dict[:σs])
    end
    args_dict[:Dg_h] = parse_csv(Int, args_dict[:Dg_h])
    args_dict[:Df_h] = parse_csv(Int, args_dict[:Df_h])

    # Convert dict to named tuple
    args = dict2namedtuple(args_dict)
    
    return args
end

