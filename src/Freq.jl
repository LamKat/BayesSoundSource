# using ADTypes: AutoReverseDiff, AutoForwardDiff
using LogExpFunctions

stable_euclidean(a, b) = sqrt(sum(abs2, a .- b) + 1e-12)

function toa_map(xs, toa_measured, speed_of_sound)


    N = length(toa_measured)

    f(θ) = begin
        y_xs = θ[1:N]
        y_ys = θ[(N+1):2N]
        t0s = θ[(2N+1):3N]
        ℓ = θ[3N+1]

        likelihood = 0

        for i ∈ 1:N
            delay = map(xs) do x
                d = stable_euclidean(x, [y_xs[i], y_ys[i]])
                return d / speed_of_sound
            end
            toa_estimated = delay .+ t0s[i]
            likelihood += sum((toa_estimated .- toa_measured[i]) .^ 2)
        end



        k = 100 * with_lengthscale(SEKernel(), ℓ)
        Σ = Symmetric(kernelpdmat(k, t0s) + 1e-4I)
        # Σinv = inv(Σ)

        # prior = y_xs' * Σinv * y_xs + 
        #         y_ys' * Σinv * y_ys + 
        #         logdet(Σ)

        L = cholesky(Σ)

        prior =
            dot(y_xs, L \ y_xs) +
            dot(y_ys, L \ y_ys) +
            logdet(L)


        σ² = 0.0005^2
        return (1/σ²) * likelihood + prior
    end

    result = optimize(
        f,
        vcat(zeros(Float64, N), zeros(Float64, N), zeros(Float64, N), 1.0),
        BFGS(); # quasi-Newton
        # autodiff = AutoForwardDiff()
    )

    return result.minimizer, result.minimum
end



function toa_sample_error(toa_sets, estimates, receivers, speed_of_sound)
    map(zip(toa_sets, estimates)) do (toa_measured, y_estimate)
        d = map(x -> BaySSL.euclidean(x, y_estimate), receivers)
        d = d .- d[2]
        τs = d ./ speed_of_sound
        residuals = (abs.(τs .- toa_measured))
        return round.(residuals .* 1000, digits=2)
    end
end


function leastsquares_toa(toa_set, receiver_locations, x0, speed_of_sound)
    optim_options = Optim.Options(show_trace=false, time_limit=100)

    lower = [-100.0, -100.0, 0.0]
    upper = [100.0, 100.0, 100.0]

    results = optimize(lower, upper, x0, Fminbox(GradientDescent()), optim_options) do coord
        d = map(x -> BaySSL.euclidean(x, coord), receiver_locations)
        d = d .- d[2]
        τs = d ./ speed_of_sound
        sum((τs .- toa_set) .^ 2)
    end
    return results.minimizer, results.minimum
end





function tdoa_sample_error(tdoa_sets, estimates, receivers, speed_of_sound)
    map(zip(tdoa_sets, estimates)) do (tdoa_measured, y_estimate)
        d = map(x -> BaySSL.euclidean(x, y_estimate), receivers)
        map(tdoa_measured) do tdoa_obs
            τ = (d[tdoa_obs[1]] - d[tdoa_obs[2]]) / speed_of_sound
            residuals = abs(τ - tdoa_obs[3])
            return round.(residuals .* 1000, digits=2)
        end
    end
end



function leastsquares_tdoa(tdoa_set, receiver_locations, x0, speed_of_sound)
    optim_options = Optim.Options(show_trace=false, time_limit=100)

    lower = [-100.0, -100.0, 0.0]
    upper = [100.0, 100.0, 100.0]

    results = optimize(lower, upper, x0, Fminbox(GradientDescent()), optim_options) do coord
        sum(tdoa_set) do (to_id, from_id, delay)
            dist_i = euclidean(coord, receiver_locations[from_id])
            dist_j = euclidean(coord, receiver_locations[to_id])
            tdoa = (dist_j - dist_i) / speed_of_sound
            return (tdoa - delay)^2
        end
    end
    return results.minimizer, results.minimum
end

import Optim: optimize, BFGS, minimizer
# using LinearAlgebra

function tdoa_mle(xs, tdoa_set, speed_of_sound; y_init=[0.0, 0.0, 0.0])

    Q = differences_matrix(length(xs))

    fixlogz(y) = [y[1], y[2], exp(y[3])]

    f(y_) = begin
        y = fixlogz(y_)
        tdoa_est = Q * [euclidean(y, x) / speed_of_sound for x ∈ xs]
        sum((tdoa_est .- tdoa_set) .^ 2)
    end

    result = optimize(
        f,
        y_init,
        BFGS()              # quasi-Newton
        # autodiff = :forward  # automatic differentiation
    )

    return fixlogz(result.minimizer), result.minimum
end


using Optim: optimize, GradientDescent, BFGS, minimizer, Fminbox

function tdoa_mle(xs, tdoa_set, speed_of_sound; y_init=[0.0, 0.0, 0.0])

    Q = differences_matrix(length(xs))

    fixlogz(y) = [y[1], y[2], exp(y[3])]

    f(θ) = begin
        y = fixlogz(θ)
        tdoa_est = Q * [euclidean(y, x) / speed_of_sound for x ∈ xs]
        sum((tdoa_est .- tdoa_set) .^ 2)
    end

    result = optimize(
        f,
        y_init,
        BFGS(), # quasi-Newton
        # Optim.Options(iterations = 10_000)
    )

    return fixlogz(result.minimizer), result.minimum
end


function toa_mle(xs, toa_set, speed_of_sound; y_init=[0.0, 0.0, 0.0, 0.0])
    fixlogz(y) = [y[1], y[2], exp(y[3])]

    f(θ) = begin
        y = fixlogz(θ[1:3])
        t₀ = θ[4]
        delay = [euclidean(y, x) / speed_of_sound for x ∈ xs]
        toa_est = delay .+ t₀
        sum((toa_est .- toa_set) .^ 2)
    end

    result = optimize(
        f,
        y_init,
        BFGS(); # quasi-Newton
    )

    return fixlogz(result.minimizer[1:3]), result.minimizer[4], result.minimum
end

function tdoa_mle_2d(xs, tdoa_measured, speed_of_sound; y_init=[0.0, 1.0, 0.0], lower=[-30, 0, -Inf], upper=[30, 30, Inf])
    stable_euclidean(a, b) = sqrt(sum(abs2, a .- b) + 1e-12)
    Q = differences_matrix(length(xs))

    f(θ) = begin
        y = θ[1:2]
        delay = map(xs) do x
            d = stable_euclidean(x, y)
            return d / speed_of_sound
        end
        tdoa_estimated = Q * delay
        sum((tdoa_estimated .- tdoa_measured) .^ 2)
    end

    result = optimize(
        f,
        lower,
        upper,
        y_init,
        Fminbox(BFGS()); # quasi-Newton
    )

    return result.minimizer[1:2], result.minimum
end


function toa_mle_2d(
    xs,
    toa_measured,
    speed_of_sound;
    y_init=[0.0, 1.0, 0.0],
    lower=[-30, 0, -Inf],
    upper=[30, 30, Inf])
    stable_euclidean(a, b) = sqrt(sum(abs2, a .- b) + 1e-12)
    f(θ) = begin
        y = θ[1:2]
        t₀ = θ[3]
        delay = map(xs) do x
            d = stable_euclidean(x, y)
            return d / speed_of_sound
        end
        toa_estimated = delay .+ t₀
        sum((toa_estimated .- toa_measured) .^ 2)
    end

    result = optimize(
        f,
        lower,
        upper,
        y_init,
        Fminbox(BFGS()); # quasi-Newton
    )

    return result.minimizer[1:2], result.minimizer[3], result.minimum
end

