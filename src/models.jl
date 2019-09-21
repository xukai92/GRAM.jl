using DensityRatioEstimation: pairwise_sqd, gaussian_gram_by_pairwise_sqd

function compute_mmd(x_de, x_nu; σs=[], verbose=false)
    function _compute_mmd_sq(pdot_dede, pdot_denu, pdot_nunu, σ)
        Kdede = gaussian_gram_by_pairwise_sqd(pdot_dede, σ)
        Kdenu = gaussian_gram_by_pairwise_sqd(pdot_denu, σ)
        Knunu = gaussian_gram_by_pairwise_sqd(pdot_nunu, σ)
        return mean(Kdede) - 2mean(Kdenu) + mean(Knunu)
    end
    mmd_sq = multi_run(_compute_mmd_sq, x_de, x_nu, σs, verbose)
    return sqrt(mmd_sq + 1f-6)
end

function estimate_ratio(x_de, x_nu; σs=[], verbose=false)
    function _estimate_ratio(pdot_dede, pdot_denu, pdot_nunu, σ)
        Kdede = gaussian_gram_by_pairwise_sqd(pdot_dede, σ)
        Kdenu = gaussian_gram_by_pairwise_sqd(pdot_denu, σ)
        n_de, n_nu = size(Kdenu)
        return convert(Float32, n_de / n_nu) * ((Kdede + diagm(0 => fill(1f-3 * one(Float32), size(Kdede, 1)))) \ sum(Kdenu; dims=2)[:,1])
    end
    return multi_run(_estimate_ratio, x_de, x_nu, σs, verbose) / convert(Float32, length(σs))
end

# TODO: test against TF implementation from Akash

# TODO: implement running average of median
function multi_run(f_run, x_de, x_nu, σs, verbose)
    pdot_dede = pairwise_sqd(x_de)
    pdot_denu = pairwise_sqd(x_de, x_nu)
    pdot_nunu = pairwise_sqd(x_nu)
    
    if isempty(σs)
        σ = sqrt(median(vcat(vec.(Flux.data.([pdot_dede, pdot_denu, pdot_nunu])))))
        if verbose
            @info "Automatically choose σ using the median of pairwise distances: $σ."
        end
        @tb @info "train" σ_median=σ log_step_increment=0
        σs = [σ]
    end

    return mapreduce(σ -> f_run(pdot_dede, pdot_denu, pdot_nunu, σ), +, σs)
end

###

abstract type AbstractGenerativeModel end

function update_by_loss!(loss, ps, opt)
    gs = Tracker.gradient(() -> loss, ps)
    Tracker.update!(opt, ps, gs)
end

function train!(m::AbstractGenerativeModel, n_epochs::Int, dl::DataLoader)
    with_logger(m.logger) do
        @showprogress for epoch in 1:n_epochs, (x_data,) in dl.train
            # Step training
            step_info = step!(m, x_data)
            m.iter.x += 1
            # Logging
            Flux.testmode!(m)
            @info "train" step_info...
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
    ps_g
    opt
    σs
end

Flux.mapchildren(f, m::MMDNet) = MMDNet(m.iter, m.logger, f(m.g), m.ps_g, m.opt, m.σs)
Flux.children(m::MMDNet) = (m.iter, m.logger, m.g, m.ps_g, m.opt, m.σs)

function step!(m::MMDNet, x_data)
    Flux.testmode!(m, false)
    x_gen = rand(m.g)
    loss_g = compute_mmd(x_gen, x_data; σs=m.σs)
    update_by_loss!(loss_g, m.ps_g, m.opt)

    return (
        loss_g=loss_g, 
        batch_size=last(size(x_data)), 
        batch_size_gen=last(size(x_gen)),
        lr=m.opt.eta,
    )
end

vis(d::Data, m::MMDNet) = vis(d, m.g)

# RMMMDNet

struct RMMMDNet <: AbstractGenerativeModel
    iter::Base.RefValue{Int}
    logger::AbstractLogger
    g::Generator
    ps_g
    f::Projector
    ps_f
    opt
    σs
end

Flux.mapchildren(f, m::RMMMDNet) = RMMMDNet(m.iter, m.logger, f(m.g), m.ps_g, f(m.f), m.ps_f, m.opt, m.σs)
Flux.children(m::RMMMDNet) = (m.iter, m.logger, m.g, m.ps_g, m.f, m.ps_f, m.opt, m.σs)

function step!(m::RMMMDNet, x_data)
    # Train f
    Flux.testmode!(m.f, false)
    Flux.testmode!(m.g, true)
    # Sample from generator
    x_gen = rand(m.g)
    # Step projector
    fx_gen, fx_data = m.f(x_gen), m.f(x_data)
    ratio = estimate_ratio(fx_gen, fx_data; σs=m.σs)
    loss_f = -mean(ratio)
    update_by_loss!(loss_f, m.ps_f, m.opt)

    # Train g
    Flux.testmode!(m.f, true)
    Flux.testmode!(m.g, false)
    # Sample from generator
    x_gen = rand(m.g)
    # Step generator
    fx_gen, fx_data = m.f(x_gen), m.f(x_data)
    loss_g = compute_mmd(fx_gen, fx_data; σs=m.σs)
    update_by_loss!(loss_g, m.ps_g, m.opt)

    return (
        loss_f=loss_f, loss_g=loss_g, 
        batch_size=last(size(x_data)), 
        batch_size_gen=last(size(x_gen)),
        lr=m.opt.eta,
    )
end

vis(d::Data, m::RMMMDNet) = vis(d, m.g)
