abstract type AbstractGenerativeModel end

function res2logdict(res::NamedTuple)
    multipliers = map(n -> Symbol("$(n)_multiplier") => (length(res[n]) > 1 ? first(res[n]) : 1), keys(res))
    return Dict((n => Flux.data(last(res[n])) for n in keys(res))..., multipliers...)
end

function train!(m::AbstractGenerativeModel, n_epochs::Int, dl::DataLoader, batch_size_gen::Int)
    with_logger(m.logger) do
        ps = Flux.params(m)
        @showprogress for epoch in 1:n_epochs, (x_data,) in dl.train
            # Run model
            Flux.testmode!(m, false)
            res = train_forward(m, x_data, batch_size_gen)
            loss = sum(prod.(values(res)))
            # Update model parameters
            gs = Tracker.gradient(() -> loss, ps)
            update!(m, gs)
            m.iter.x += 1
            # Logging
            Flux.testmode!(m)
            logdict = Dict(
                res2logdict(res)...,
                :loss => Flux.data(loss), 
                :lr => m.opt.eta,
                :batch_size => size(x_data, 2), 
                :batch_size_gen => batch_size_gen,
            )
            @info "train" logdict...
            if m.iter.x % 10 == 0
                @info "eval" gen=vis(dl.data, m) log_step_increment=0
            end
        end
    end
end

# MMDNet

struct MMDNet <: AbstractGenerativeModel
    iter::Base.RefValue{Int}
    logger::AbstractLogger
    g::Generator
    opt
    σs
end

Flux.mapchildren(f, m::MMDNet) = MMDNet(m.iter, m.logger, f(m.g), m.opt, m.σs)
Flux.children(m::MMDNet) = (m.iter, m.logger, m.g, m.opt, m.σs)

function train_forward(m::MMDNet, x_data, batch_size_gen::Int)
    x_gen = rand(m.g, batch_size_gen)
    mmd = estimate_ratio_and_compute_mmd(x_gen, x_data; σs=m.σs).mmd
    return (loss_g=mmd,)
end

function update!(m::MMDNet, gs)
    ps_g = Flux.params(m.g)
    Tracker.update!(m.opt, ps_g, gs)
end

vis(d::Data, m::MMDNet) = vis(d, m.g)

# RMMMDNet

struct RMMMDNet <: AbstractGenerativeModel
    iter::Base.RefValue{Int}
    logger::AbstractLogger
    g::Generator
    f::Projector
    opt
    σs
end

Flux.mapchildren(f, m::RMMMDNet) = RMMMDNet(m.iter, m.logger, f(m.g), f(m.f), m.opt, m.σs)
Flux.children(m::RMMMDNet) = (m.iter, m.logger, m.g, m.f, m.opt, m.σs)

function train_forward(m::RMMMDNet, x_data, batch_size_gen::Int)
    x_gen = rand(m.g, batch_size_gen)
    fx_gen, fx_data = m.f(x_gen), m.f(x_data)
    ratio, mmd = estimate_ratio_and_compute_mmd(fx_gen, fx_data; σs=m.σs)
#     return (loss_g=mmd, loss_f=(1, -mean((ratio .- 1) .^ 2)))
    return (loss_g=mmd, loss_f=(1, -mean(ratio)))
end

function update!(m::RMMMDNet, gs)
    ps_g = Flux.params(m.g)
    Tracker.update!(m.opt, ps_g, gs)
    ps_f = Flux.params(m.f)
    Tracker.update!(m.opt, ps_f, gs)
end

vis(d::Data, m::RMMMDNet) = vis(d, m.g)
