__precompile__(false)
module RMMMDNets

using Statistics, LinearAlgebra, StatsFuns, Distributions, Humanize, Dates, Flux.Data.MNIST
using Random: MersenneTwister, shuffle

using MLToolkit
using MLToolkit.Neural: IntIte, Tracker
import MLToolkit.Neural: evaluate, update!
import MLToolkit: parse_toml, process_argdict, NeuralSampler
export DataLoader

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

function NeuralSampler(
    base, 
    Dz::Int, 
    Dhs::IntIte, 
    Dx::Int, 
    σ::Function, 
    σlast::Function, 
    isnorm::Bool, 
    n_default::Int
)
    size(base) != (Dz,) && throw(DimensionMismatch("size(base) ($(size(base))) != (Dz,) ($((Dz,)))"))
    return NeuralSampler(base, DenseNet(Dz, Dhs, Dx, σ, σlast; isnorm=isnorm), n_default)
end

### Projector

function DenseProjector(Dx::Int, Dhs::IntIte, Df::Int, σ::Function, isnorm::Bool)
    return Projector(DenseNet(Dx, Dhs, Df, σ, identity; isnorm=isnorm), Df)
end

function ConvProjector(Dx::Int, Df::Int, σ::Function, isnorm::Bool)
    @assert Dx == 28 * 28 "[ConvProjector] Only MNIST-like data is supported."
    return Projector(ConvNet((28, 28, 1), Df, σ, identity; isnorm=isnorm), Df)
end

### Discriminator

function DenseDiscriminator(Dx::Int, Dhs::IntIte, σ::Function, isnorm::Bool)
    return Discriminator(DenseNet(Dx, Dhs, 1, σ, sigmoid; isnorm=isnorm))
end

function ConvDiscriminator(Dx::Int, σ::Function, isnorm::Bool)
    @assert Dx == 28 * 28 "[ConvDiscriminator] Only MNIST-like data is supported."
    return Discriminator(ConvNet((28, 28, 1), 1, σ, sigmoid; isnorm=isnorm))
end

include("models.jl")
export GAN, MMDNet, RMMMDNet, train!, evaluate

###

function get_data(name::String)
    rng = MersenneTwister(1)
    if name == "mnist"
        Xtrain = Array{Float32,2}(hcat(float.(reshape.(MNIST.images(), :))...))
    end
    if name == "gaussian"
        N = 1_000
        Xtrain = rand(rng, MvNormal(zeros(Float32, 2), Float32[0.25 0.25; 0.25 1]), N)
    end
    if name == "ring"
        N = 1_000
        Xtrain = rand(rng, Ring(8, 2, 2f-1), N)
    end
    return Dataset(gpu(Xtrain); name=name)
end

function plot!(d::Dataset, g::NeuralSampler)
    rng = MersenneTwister(1)
    if d.name == "mnist" 
        Xdata = d.train |> cpu
        Xgen = rand(rng, g, 32) |> cpu |> Tracker.data
        MLToolkit.plot!(GrayImages(hcat(Xdata[:,shuffle(rng, 1:last(size(Xdata)))[1:32]], Xgen)))
    elseif d.name in ["gaussian", "ring"]
        Xdata = d.train |> cpu
        Xgen = rand(rng, g, last(size(Xdata))) |> cpu |> Tracker.data
        plt.scatter(Xdata[1,:], Xdata[2,:], marker=".", label="data", alpha=0.5)
        plt.scatter(Xgen[1,:],  Xgen[2,:],  marker=".", label="gen",  alpha=0.5)
        autoset_lim!(Xdata)
        plt.legend(fancybox=true, framealpha=0.5)
    end
end

function plot!(d::Dataset, g::NeuralSampler, f::Projector)
    rng = MersenneTwister(1)
    if d.name == "mnist" 
        Xdata = d.train |> cpu
        Xgen = rand(rng, g, 32) 
        fXdata = f(Xdata) |> cpu |> Tracker.data
        fXgen = f(Xgen) |> cpu |> Tracker.data
        MLToolkit.plot!(GrayImages(hcat(fXdata[:,shuffle(rng, 1:last(size(fXdata)))[1:32]], fXgen)))
    elseif d.name in ["gaussian", "ring"]
        Xdata = d.train |> cpu
        Xgen = rand(rng, g, last(size(Xdata)))
        fXdata = f(Xdata) |> cpu |> Tracker.data
        fXgen = f(Xgen) |> cpu |> Tracker.data
        plt.scatter(fXdata[1,:], fXdata[2,:], marker=".", label="data", alpha=0.5)
        plt.scatter(fXgen[1,:],  fXgen[2,:],  marker=".", label="gen",  alpha=0.5)
        autoset_lim!(fXdata)
        plt.legend(fancybox=true, framealpha=0.5)
    end
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
    Dhs_g = parse_csv(Int, args.Dhs_g)
    act = parse_op(args.act)
    actlast = parse_op(args.actlast)
    g = NeuralSampler(base, args.Dz, Dhs_g, dim(dataset), act, actlast, args.norm, args.batchsize_g)
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
                f = ConvProjector(dim(dataset), args.Df, act, args.norm)
            else
                Dhs_f = parse_csv(Int, args.Dhs_f)
                f = DenseProjector(dim(dataset), Dhs_f, args.Df, act, args.norm)
            end
            m = RMMMDNet(logger, opt, sigma, g, f) |> track
        end
    end
    @info "Init $(args.modelname) with $(nparams(m) |> Humanize.digitsep) parameters" logdir
    return m |> gpu
end

function run_exp(args; model=nothing)
    seed!(args.seed)
    
    data = get_data(args.dataset)
    dataloader = DataLoader(data, args.batchsize)

    model = isnothing(model) ? get_model(args, data) : model

    with_logger(model.logger) do
        train!(model, dataloader, args.n_epochs; evalevery=100)
    end
    evaluate(model, dataloader)
    
    modelpath = savemodel(model)
    
    return dataloader, model, modelpath
end

export get_data, get_model, run_exp

end # module
