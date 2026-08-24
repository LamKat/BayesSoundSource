using LinearAlgebra
using Random
using Distributions
# using Plots


"""
    distance(a, b)

Compute the Euclidean distance between vectors `a` and `b`.
"""
@inline distance(a, b) = norm(a .- b)

function pairwise_difference_matrix(n::Int)
    pairs = [(i, j) for i in 1:n-1 for j in i+1:n]
    return [i == p[1] ? 1 : i == p[2] ? -1 : 0 for p in pairs, i in 1:n]
end

num_pairwise_differences(n::Int) = n * (n - 1) ÷ 2


"""
    synthetic_tdoa(positions, receivers, c, σ; seed=nothing)

Generate synthetic time-difference-of-arrival (TDOA) observations for a
sequence of source positions. Independent Gaussian noise 
with standard deviation `σ` is added to each observation.

Returns one TDOA observation vector per source position.
"""
function synthetic_tdoa(positions, receivers, c, σ; seed=nothing) 
    
    if seed !== nothing
        Random.seed!(seed)
    end

    N = length(receivers)

    Q = pairwise_difference_matrix(N)

    QN = num_pairwise_differences(N)

    ε = Normal(0, σ)

    map(positions) do source
        delays = [distance(source, receiver) / c for receiver ∈ receivers] 

        return Q * delays + rand(ε, QN)
    end 
end 

"""
    synthetic_toa(positions, event_times, receivers, c, σ; seed=nothing)

Generate synthetic time-of-arrival (TOA) observations for a sequence of
source positions and corresponding emission times. Independent Gaussian noise 
with standard deviation `σ` is added to each observation.

Returns one TOA observation vector per source position.
"""
function synthetic_toa(positions, event_times, receivers, c, σ; seed=nothing)
    @assert(length(positions) == length(event_times))

    if seed !== nothing
        Random.seed!(seed)
    end

    N = length(receivers)

    ε = Normal(0, σ)

    map(zip(positions, event_times)) do (source, time)
        delays = [distance(source, receiver) / c for receiver ∈ receivers] 

        time .+ delays + rand(ε, N) 
    end 
end 



"""
    stochastic_trajectory(amax; d=3, tmax=1.0, Δt=0.01,
                          x0=zeros(d), v0=zeros(d), seed=nothing)

Simulate a trajectory with randomly varying acceleration.

Returns `(times, positions)`, where `times` is the simulation time grid and
`positions` contains the simulated position at each time step.
"""
function stochastic_trajectory(amax; d = 3, tmax=1.0, Δt=0.01,
                               x0=zeros(d), v0=zeros(d), seed=nothing)

    if seed !== nothing
        Random.seed!(seed)
    end

    steps = Int(round(tmax/Δt))
    times = range(0, step=Δt, length=steps+1)

    x = x0
    v = v0

    positions = Vector{Vector{Float64}}(undef, steps+1)

    positions[1] = x

    for i in 1:steps
        # Random acceleration direction
        dir = rand(Uniform(-1, 1), d)
        dir /= norm(dir)

        a = rand(Uniform(0, amax))

        acc = a * dir
        
        # Semi-implicit Euler method
        v += acc * Δt
        x += v * Δt

        positions[i+1] = x
    end

    return times, positions
end


function rotate(p)
    anim = Animation()
    plot!(p, legend=false)
    for i in range(0, stop = 359, step = 10)
        fram = Plots.plot(p, camera=(i, 10))
        frame(anim, fram)
    end
    gif(anim, fps=10)
end 


