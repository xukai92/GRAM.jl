struct Ring{T<:AbstractFloat} <: ContinuousMultivariateDistribution
    n_clusters::Int
    s::Int
    σ::T
end

function make_ringmixture(ring::Ring)
    π_typed = convert(typeof(ring.σ), π)
    cluster_indices = collect(0:ring.n_clusters-1)
    base_angle = π_typed * 2 / ring.n_clusters
    angle = (base_angle .* cluster_indices) .- π_typed / 2
    μ = [ring.s * cos.(angle) ring.s * sin.(angle)]'
    return MixtureModel([MvNormal(μ[:,i], ring.σ) for i in 1:size(μ, 2)])
end

Distributions.rand(rng::AbstractRNG, ring::Ring, n::Int) = convert.(typeof(ring.σ), rand(rng, make_ringmixture(ring), n))

Distributions.logpdf(ring::Ring, x) = logpdf(make_ringmixture(ring), x)

###

import MLDataUtils

struct Data
    dataset::String
    train
    test
    validation
    function Data(dataset::String, train, test=nothing, validation=nothing)
        if !isnothing(test)
            @assert Base.front(train) == Base.front(test)
        end
        if !isnothing(validation)
            @assert Base.front(train) == Base.front(validation)
        end
        return new(dataset, train, test, validation)
    end
end

function Base.getproperty(d::Data, k::Symbol)
    if k == :dim
        dim = Base.front(size(d.train))
        dim = length(dim) == 1 ? dim[1] : dim
        return dim
    else
        getfield(d, k)
    end
end

struct DataLoader
    data::Data
    batch_size::Int
    batch_size_eval::Int
    function DataLoader(data::Data, batch_size::Int, batch_size_eval::Int=batch_size)
        new(data, batch_size, batch_size_eval)
    end
end
            
function Base.getproperty(dl::DataLoader, k::Symbol)
    if k in (:train, :test, :validation)
        data = getproperty(dl.data, k)
        batch_size = k == :train ? dl.batch_size : dl.batch_size_eval
        idcs_iterator = Iterators.partition(MLDataUtils.shuffleobs(1:last(size(data))), batch_size)
        return ((batch=data[:,idcs], idcs=idcs) for idcs in idcs_iterator)
    else
        getfield(dl, k)
    end
end

