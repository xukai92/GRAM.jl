__precompile__(false)
module RMMMDNets

using Statistics, LinearAlgebra, StatsFuns, Distributions, Humanize, Dates
using MLDatasets: CIFAR10, MNIST
using Random: MersenneTwister, shuffle
using Reexport: @reexport

@reexport using MLToolkit
using MLToolkit.Neural: IntIte
import MLToolkit.Neural:  evaluate, update!
import MLToolkit: parse_toml, process_argdict, NeuralSampler, plot!
export DataLoader

Flux.Zygote.@nograd Flux.conv_transpose_dims
Flux.Zygote.@nograd Flux.gpu

### Scripting

function parse_toml(hyperpath::String, dataset::String, modelname::String)
    return parse_toml(hyperpath, (:dataset => dataset, :modelname => modelname))
end

function process_argdict(argdict; override=NamedTuple(), suffix="")
    return process_argdict(
        argdict; 
        override=override,
        nameexclude=[:dataset, :modelname, :n_epochs, :actlast],
        nameinclude_last=:seed,
        suffix=suffix
    )
end

export parse_toml, process_argdict

### Neural sampler

using MLToolkit.Neural: optional_BatchNorm

# Flux.trainable(c::Conv) = (c.weight,)
# Flux.trainable(ct::ConvTranspose) = (ct.weight,)

function build_conv_outmnist(Din::Int, σs; isnorm::Bool=false)
    @assert length(σs) == 4 "Length of `σs` must be `4` for `build_conv_outmnist`"
    return Chain(
        #     Din x B
        Dense(Din,  1024), optional_BatchNorm(1024, σs[1], isnorm),
        #    1024 x B
        Dense(1024, 6272),
        # -> 6272 x B
        x -> reshape(x, 7, 7, 128, last(size(x))), optional_BatchNorm(128, σs[2], isnorm),
        # ->  7 x  7 x 128 x B
        ConvTranspose((4, 4), 128 => 64; stride=(2, 2), pad=(1, 1)), optional_BatchNorm(64, σs[3], isnorm),
        # -> 14 x 14 x  64 x B
        ConvTranspose((4, 4),  64 =>  1; stride=(2, 2), pad=(1, 1)), x -> σs[4].(x)
        # -> 28 x 28 x 1 x B
    )
end

build_conv_outmnist(Din::Int, σ::Function, σlast::Function; kwargs...) = build_conv_outmnist(Din, (σ, σ, σ, σlast); kwargs...)

function build_convnet_outcifar(Din::Int, σs; isnorm::Bool=false)
    @assert length(σs) == 4 "Length of `σs` must be `4` for `build_convnet_outcifar10`"
    return Chain(
        #     Din x B
        Dense(Din,  2048), 
        # -> 2048 x B
        x -> reshape(x, 4, 4, 128, size(x, 2)), optional_BatchNorm(128, σs[1], isnorm; momentum=9f-1),
        # ->  4 x  4 x 128 x B
        ConvTranspose((4, 4), 128 => 64; stride=(2, 2), pad=(1, 1)), optional_BatchNorm(64, σs[2], isnorm; momentum=9f-1),
        # ->  8 x  8 x  64 x B        
        ConvTranspose((4, 4),  64 => 32; stride=(2, 2), pad=(1, 1)), optional_BatchNorm(32, σs[3], isnorm; momentum=9f-1),
        # -> 16 x 16 x  64 x B
        ConvTranspose((4, 4),  32 =>  3; stride=(2, 2), pad=(1, 1)), x -> σs[4].(x)
        # -> 32 x 32 x   3 x B
    )
end

build_convnet_outcifar(Din::Int, σ::Function, σlast::Function; kwargs...) = build_convnet_outcifar(Din, (σ, σ, σ, σlast); kwargs...)

# TODO: unify building conv archs
function NeuralSampler(
    base, 
    Dz::Int, 
    Dhs::Union{IntIte,String}, 
    Dx::Union{Int,Tuple{Int,Int,Int}}, 
    σ::Function, 
    σlast::Function, 
    isnorm::Bool, 
    n_default::Int
)
    size(base) != (Dz,) && throw(DimensionMismatch("size(base) ($(size(base))) != (Dz,) ($((Dz,)))"))
    if Dx == (32, 32, 3) && Dhs == "conv"
        f = build_convnet_outcifar(Dz, σ, σlast; isnorm=isnorm)
    elseif Dx == (28, 28, 1) && Dhs == "conv"
        f = build_conv_outmnist(Dz, σ, σlast; isnorm=isnorm)
    else
        f = DenseNet(Dz, Dhs, Dx, σ, σlast; isnorm=isnorm)
    end
    return NeuralSampler(base, f, n_default)
end

### Projector

function DenseProjector(Dx::Int, Dhs::IntIte, Df::Int, σ::Function, isnorm::Bool)
    return Projector(DenseNet(Dx, Dhs, Df, σ, identity; isnorm=isnorm), Df)
end

function ConvProjector(Dx::Union{Int,Tuple{Int,Int,Int}}, Df::Int, σ::Function, isnorm::Bool)
    if Dx == 28 * 28 || Dx == (28, 28, 1)
        return Projector(ConvNet((28, 28, 1), Df, σ, identity; isnorm=isnorm), Df)
    elseif Dx == (32, 32, 3)
        return Projector(ConvNet((32, 32, 3), Df, σ, identity; isnorm=isnorm), Df)
    else
        @error "[ConvProjector] Only MNIST-like or CIFAR-like data is supported."
    end
end

### Discriminator

function DenseDiscriminator(Dx::Int, Dhs::IntIte, σ::Function, isnorm::Bool)
    return Discriminator(DenseNet(Dx, Dhs, 1, σ, sigmoid; isnorm=isnorm))
end

function ConvDiscriminator(Dx::Union{Int,Tuple{Int,Int,Int}}, σ::Function, isnorm::Bool)
    if Dx == 28 * 28
        return Discriminator(ConvNet((28, 28, 1), 1, σ, sigmoid; isnorm=isnorm))
    elseif Dx == 32 * 32 * 3
        return Discriminator(ConvNet((32, 32, 3), 1, σ, sigmoid; isnorm=isnorm))
    else
        @error "[ConvDiscriminator] Only MNIST-like or CIFAR-like data is supported."
    end
end

include("models.jl")
export GAN, MMDNet, RMMMDNet, train!, evaluate

###

flatten(x) = reshape(x, :, last(size(x)))

function get_data(name::String)
    rng = MersenneTwister(1)
    if name == "mnist"
        Xtrain = permutedims(MNIST.traintensor(Float32), (2, 1, 3))
        Xtrain = flatten(Xtrain)
#         Xtrain = reshape(Xtrain, (28, 28, 1, last(size(Xtrain))))
    end
    if name == "cifar10"
        Xtrain = permutedims(CIFAR10.traintensor(Float32), (2, 1, 3, 4))
        Xtrain = Xtrain .* 2 .- 1
    end
    if name == "gaussian"
        N = 1_000
        Xtrain = rand(rng, MvNormal(zeros(Float32, 2), Float32[0.25 0.25; 0.25 1]), N)
    end
    if name == "ring"
        N = 1_000
        Xtrain = rand(rng, Ring(8, 2, 2f-1), N)
    end
    Xtrain = Xtrain |> gpu
    return Dataset(Xtrain; name=name)
end

function plot!(d::Dataset, g::NeuralSampler)
    rng = MersenneTwister(1)
    if d.name == "mnist"
        Xdata = d.train
        Xgen = rand(rng, g, 32) |> cpu
        MLToolkit.plot!(GrayImages(cat(Xdata[:,shuffle(rng, 1:last(size(Xdata)))[1:32]], Xgen; dims=2)))
#         MLToolkit.plot!(GrayImages(dropdims(cat(Xdata[:,:,:,shuffle(rng, 1:last(size(Xdata)))[1:32]], Xgen; dims=4); dims=3)))
    elseif d.name == "cifar10"
        Xdata = d.train
        Xgen = rand(rng, g, 32) |> cpu
        Xshow = cat(Xdata[:,:,:,shuffle(rng, 1:last(size(Xdata)))[1:32]], Xgen; dims=4)
        MLToolkit.plot!(RGBImages((Xshow .+ 1) ./ 2))
    elseif d.name in ["gaussian", "ring"]
        Xdata = d.train
        Xgen = rand(rng, g, last(size(Xdata))) |> cpu
        plt.scatter(Xdata[1,:], Xdata[2,:], marker=".", label="data", alpha=0.5)
        plt.scatter(Xgen[1,:],  Xgen[2,:],  marker=".", label="gen",  alpha=0.5)
        autoset_lim!(Xdata)
        plt.legend(fancybox=true, framealpha=0.5)
    end
end

function plot!(d::Dataset, g::NeuralSampler, f::Projector)
    rng = MersenneTwister(1)
    Xdata = d.train
    Xgen = rand(rng, g, last(size(Xdata)))
    fXdata = f(Xdata) |> cpu
    fXgen = f(Xgen) |> cpu
    plt.scatter(fXdata[1,:], fXdata[2,:], marker=".", label="data", alpha=0.5)
    plt.scatter(fXgen[1,:],  fXgen[2,:],  marker=".", label="gen",  alpha=0.5)
    autoset_lim!(fXdata)
    plt.legend(fancybox=true, framealpha=0.5)
end

parse_csv(T, l) = map(x -> parse(T, x), split(l, ","))
parse_op(op::String) = eval(Symbol(op))
parse_op(op) = op

function get_model(args::NamedTuple, dataset::Dataset)
    module_path = pathof(@__MODULE__) |> splitdir |> first |> splitdir |> first
    logdir = "$(dataset.name)/$(args.modelname)/$(args.expname)/$(Dates.format(now(), DATETIME_FMT))"
    logger = TBLogger("$module_path/logs/$logdir")
    if args.opt == "adam"
        opt = ADAM(args.lr, (args.beta1, 999f-4))
    elseif args.opt == "rmsprop"
        opt = RMSProp(args.lr)
    end
    if args.base == "uniform"
        base = UniformNoise(args.Dz)
    elseif args.base == "gaussian"
        base = GaussianNoise(args.Dz)
    end
    if args.Dhs_g == "conv"
        if :act in keys(args)
            @warn "args.act is ignored"
        end
        if :actlast in keys(args)
            @warn "args.actlast is ignored"
        end
        actlast = dataset.name == "cifar10" ? tanh : sigmoid
        g = NeuralSampler(base, args.Dz, "conv", dim(dataset), relu, actlast, args.norm, args.batchsize_g)
    else
        Dhs_g = parse_csv(Int, args.Dhs_g)
        act = parse_op(args.act)
        actlast = parse_op(args.actlast)
        g = NeuralSampler(base, args.Dz, Dhs_g, dim(dataset), act, actlast, args.norm, args.batchsize_g)
    end
    if args.modelname == "gan"
        if args.Dhs_d == "conv"
            d = ConvDiscriminator(dim(dataset), act, args.norm)
        else
            Dhs_d = parse_csv(Int, args.Dhs_d)
            d = DenseDiscriminator(dim(dataset), Dhs_d, act, args.norm)
        end
        m = GAN(logger, opt, g, d)
    else
        sigma = args.sigma == "median" ? [] : parse_csv(Float32, args.sigma)
        if args.modelname == "mmdnet"
            m = MMDNet(logger, opt, sigma, g)
        elseif args.modelname == "rmmmdnet"
            if args.Dhs_f == "conv"
                act = x -> leakyrelu(x, 2f-1)
                if :act in keys(args)
                    @warn "args.act is ignored"
                end
                f = ConvProjector(dim(dataset), args.Df, act, args.norm)
            else
                Dhs_f = parse_csv(Int, args.Dhs_f)
                f = DenseProjector(dim(dataset), Dhs_f, args.Df, act, args.norm)
            end
            m = RMMMDNet(logger, opt, sigma, g, f)
        end
    end
    @info "Init $(args.modelname) with $(nparams(m) |> Humanize.digitsep) parameters" logdir
    return m |> gpu
end

function run_exp(args; model=nothing, initonly=false)
    seed!(args.seed)
    
    data = get_data(args.dataset)
    dataloader = DataLoader(data, args.batchsize)

    model = isnothing(model) ? get_model(args, data) : model
    
    if !initonly
        with_logger(model.logger) do
            train!(model, dataloader, args.n_epochs; evalevery=50)
        end
    end
    
    evaluate(model, dataloader)
    
    modelpath = savemodel(model)
    
    return dataloader, model, modelpath
end

export get_data, get_model, run_exp

end # module
