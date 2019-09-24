# Copying codes for anonymization purpose

import PyPlot, PyCall, Images

# Pre-allocating Python bindings
const axes_grid1 = PyCall.PyNULL()
const plt_agg = PyCall.PyNULL()
# Matplotlib and PyPlot
const mpl = PyCall.PyNULL()
const plt = PyCall.PyNULL()

function __init__()
    copy!(axes_grid1, PyCall.pyimport("mpl_toolkits.axes_grid1"))
    copy!(mpl, PyPlot.matplotlib)
    copy!(plt, mpl.pyplot)
    copy!(plt_agg, mpl.backends.backend_agg)
end

const DATETIME_FMT = "ddmmyyyy-H-M-S"

istb() = Logging.current_logger() isa TensorBoardLogger.TBLogger

macro tb(expr)
    return esc(
        quote
            if istb()
                $expr
            end
        end
    )
end

nparams(m) = sum(prod.(size.(Flux.params(m))))

function flatten_dict(dict::Dict{T,<:Any};
    equal_sym="=",
    delimiter="-",
    exclude::Vector{T}=T[],
    include::Vector{T}=collect(keys(dict))) where {T<:Union{String,Symbol}}
    @assert issubset(Set(exclude), keys(dict)) "Keyword `exclude` must be a subset of `keys(dict)`; set diff: $(setdiff(Set(exclude), keys(dict)))"
    @assert issubset(Set(include), keys(dict)) "Keyword `include` must be a subset of `keys(dict)`; set diff: $(setdiff(Set(include), keys(dict)))"
    return join(["$k$equal_sym$v" for (k,v) in filter(t -> (t[1] in include) && !(t[1] in exclude), dict)], delimiter)
end

function dict2namedtuple(d)
    return NamedTuple{tuple(keys(d)...),typeof(tuple(values(d)...))}(tuple(values(d)...))
end

function plot_grayimg!(img, args...; ax=plt.gca())
    @assert length(args) == 0 || length(args) == 2 "You can either plot a single image or declare the `n_rows` and `n_cols`"
    im = ax."imshow"(make_imggrid(img, args...), cmap="gray")
    plt.axis("off")
    divider = axes_grid1.make_axes_locatable(ax)
    cax = divider."append_axes"("bottom", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax, orientation="horizontal")
    return ax
end

function make_imggrid(x, n_rows, n_cols; flat=true, gap::Integer=1)
    if !flat
        x = rehsape(x, size(x, 1)^2, size(x, 4))
    end
    d², n = size(x)
    d = round(Int, sqrt(d²))
    x_show = 0.5 * ones(Float32, n_rows * (d + gap) + gap, n_cols * (d + gap) + gap)
    i = 1
    for row = 1:n_rows, col = 1:n_cols
        if i <= n
            row_i = (row - 1) * (d + gap) + 1
            col_i = (col - 1) * (d + gap) + 1
            x_show[row_i+1:row_i+d,col_i+1:col_i+d] = x[:,i]
        else
            break
        end
        i += 1
    end
    return x_show
end

function make_imggrid(x; kargs...)
    n = size(x, 2)
    l = ceil(Integer, sqrt(n))
    n_rows = l * (l - 1) > n ? l - 1 : l
    return make_imggrid(x, l, n_rows; kargs...)
end

function autoset_lim!(x; ax=plt.gca())
    xlims = [extrema(x[1,:])...]
    dx = xlims[2] - xlims[1]
    xlims += [-0.1dx, +0.1dx]
    ylims = [extrema(x[2,:])...]
    dy = ylims[2] - ylims[1]
    ylims += [-0.1dy, +0.1dy]
    dim = size(x, 1)
    ax.set_xlim(xlims)
    ax.set_ylim(ylims)
end

function figure_to_image(fig::PyPlot.Figure; close=true)
    canvas = plt_agg.FigureCanvasAgg(fig)
    canvas.draw()
    data = canvas.buffer_rgba() ./ 255
    w, h = fig.canvas.get_width_height()
    img = [Images.RGBA(data[r,c,:]...) for r in 1:h, c in 1:w]
    if close plt.close(fig) end
    return img
end

TensorBoardLogger.preprocess(name, fig::PyPlot.Figure, data) = push!(data, name => figure_to_image(fig))

TensorBoardLogger.preprocess(name, x::Tracker.TrackedReal, data) = push!(data, name => Flux.data(x))

###

function pairwise_sqd(x)
    n = size(x, 2)
    xixj = x' * x
    xsq = sum(x .^ 2; dims=1)
    return repeat(xsq'; outer=(1, n)) + repeat(xsq; outer=(n, 1)) - 2xixj
end

function pairwise_sqd(x, y)
    nx = size(x, 2)
    ny = size(y, 2)
    xiyj = x' * y
    xsq = sum(x .^ 2; dims=1)
    ysq = sum(y .^ 2; dims=1)
    return repeat(xsq'; outer=(1, ny)) .+ repeat(ysq; outer=(nx, 1)) - 2xiyj
end

###

# using Flux.CuArrays

# Base.inv(x::CuArray{<:Real,2}) = CuArrays.CUBLAS.matinv_batched([x])[2][1]

# const CuOrAdj = Union{CuArray, LinearAlgebra.Adjoint{T, CuArray{T, 2}}} where {T<:AbstractFloat}
# function Base.:\(_A::AT1, _B::AT2) where {AT1<:CuOrAdj, AT2<:CuOrAdj}
#     A, B = copy(_A), copy(_B)
#     A, ipiv = CuArrays.CUSOLVER.getrf!(A)
#     return CuArrays.CUSOLVER.getrs!('N', A, ipiv, B)
# end

# Tracker.@grad function (A \ B)
#     return Tracker.data(A) \ Tracker.data(B), function (Δ)
#         ∇A = -(A' \ Δ) * B' / A'
#         return (∇A,  (A' \ Δ))
#     end
# end