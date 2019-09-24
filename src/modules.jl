# Generator

parse_op(op::String) = eval(Symbol(op))
parse_op(op)= op

function build_dense_with_norm(D_in, D_out, σ, norm::Bool=false)
    return if norm
        Chain(Dense(D_in, D_out), BatchNorm(D_out, σ))
    else
        Dense(D_in, D_out, σ)
    end
end

function build_mlp_chain(D_in, D_h, D_out, σ, σ_last; norm::Bool=false)
    n_layers = length(D_h)
    return Chain(
        build_dense_with_norm(D_in, D_h[1], σ, norm),
        [build_dense_with_norm(D_h[i], D_h[i+1], σ, norm) for i in 1:n_layers-1]...,
        Dense(D_h[n_layers], D_out, σ_last)
    )
end

# function build_conv_chain(D_in, D_h, D_out, σ, σ_last)
#     return Chain(
#         Dense(D_z, 144, relu),
#         x -> reshape(x, 3, 3, 16, last(size(x))), # ( 3,  3, 16, 1)
#         ConvTranspose((3, 3), 16 => 8, relu),     # ( 5,  5,  8, 1)
#         ConvTranspose((6, 6), 8 => 4, relu),      # (10, 10,  4, 1)
#         ConvTranspose((8, 8), 4 => 2, relu),      # (17, 17,  2, 1)
#         ConvTranspose((12, 12), 2 => 1, sigmoid), # (28, 28,  1, 1)
#         x -> reshape(x, D_x, last(size(x)))
#     )
# end

# TODO: improve the constructo by implementing `build_conv_with_norm`
function build_conv_chain(D_in, D_out, σ, σ_last; norm::Bool=false)
    @assert D_in == 28 * 28 "`build_conv_chain` is handed coded to only support MNIST-like data (28 x 28 images)."
    if norm
        return Chain(
            x -> reshape(x, 28, 28, 1, last(size(x))),                          #    28 x 28 x  1
            Conv((3, 3), 1=>16,  pad=(1,1)), BatchNorm(16, σ), MaxPool((2, 2)), # -> 14 x 14 x 16
            Conv((3, 3), 16=>32, pad=(1,1)), BatchNorm(32, σ), MaxPool((2, 2)), # ->  7 x  7 x 32
            Conv((3, 3), 32=>32, pad=(1,1)), BatchNorm(32, σ), MaxPool((2, 2)), # ->  3 x  3 x 32
            x -> reshape(x, :, size(x, 4)),
            Dense(288, D_out, σ_last)
        )
    else
        return Chain(
            x -> reshape(x, 28, 28, 1, last(size(x))),           #    28 x 28 x  1
            Conv((3, 3), 1=>16,  pad=(1,1), σ), MaxPool((2, 2)), # -> 14 x 14 x 16
            Conv((3, 3), 16=>32, pad=(1,1), σ), MaxPool((2, 2)), # ->  7 x  7 x 32
            Conv((3, 3), 32=>32, pad=(1,1), σ), MaxPool((2, 2)), # ->  3 x  3 x 32
            x -> reshape(x, :, size(x, 4)),
            Dense(288, D_out, σ_last)
        )
    end
end

abstract type AbstractBaseGenerator <: ContinuousMultivariateDistribution end

struct UniformBase <: AbstractBaseGenerator
    D_z::Int
end

function Distributions.rand(rng::AbstractRNG, b::UniformBase, n::Int)
    z = 2 * rand(rng, Float32, b.D_z, n) .- 1
    return use_gpu.x ? gpu(z) : z
end

struct GaussianBase <: AbstractBaseGenerator
    D_z::Int
end

function Distributions.rand(rng::AbstractRNG, b::GaussianBase, n::Int)
    z = randn(rng, Float32, b.D_z, n)
    return use_gpu.x ? gpu(z) : z
end

struct Generator
    base::AbstractBaseGenerator
    f
    n_default::Int
end

Flux.@treelike(Generator)

function Generator(base::AbstractBaseGenerator, D_z::Int, D_h::AbstractVector{Int}, D_x::Int, σ, σ_last, norm, n_default::Int)
    @assert base.D_z == D_z
    σ, σ_last = parse_op(σ), parse_op(σ_last)
    f = build_mlp_chain(D_z, D_h, D_x, σ, σ_last; norm=norm)
    return Generator(base, f, n_default)
end

function Distributions.rand(rng::AbstractRNG, g::Generator, n::Int=g.n_default)
    z = rand(g.base, n)
    return g.f(z)
end

Distributions.rand(g::Generator, n::Int) = rand(GLOBAL_RNG, g, n)

# Projector

struct Projector
    f
    D_fx::Int
end

Flux.@treelike(Projector)

(p::Projector)(x) = p.f(x)

function Projector(D_x::Int, D_h::AbstractArray{Int}, D_fx::Int, σ, norm)
    σ = parse_op(σ)
    f = build_mlp_chain(D_x, D_h, D_fx, σ, identity; norm=norm)
    return Projector(f, D_fx)
end

function ConvProjector(D_x::Int, D_fx::Int, σ, norm)
    σ = parse_op(σ)
    f = build_conv_chain(D_x, D_fx, σ, identity; norm=norm)
    return Projector(f, D_fx)
end

# Discriminator

struct Discriminator
    f
end

Flux.@treelike(Discriminator)

(d::Discriminator)(x) = d.f(x)

function Discriminator(D_x::Int, D_h::AbstractArray{Int}, σ, norm)
    σ = parse_op(σ)
    f = build_mlp_chain(D_x, D_h, 1, σ, identity; norm=norm)
    return Discriminator(f)
end

function ConvDiscriminator(D_x::Int, σ, norm)
    σ = parse_op(σ)
    f = build_conv_chain(D_x, 1, σ, identity; norm=norm)
    return Discriminator(f)
end
