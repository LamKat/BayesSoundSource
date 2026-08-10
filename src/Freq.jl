# using ADTypes: AutoReverseDiff, AutoForwardDiff
using LogExpFunctions

using Optim: optimize, GradientDescent, BFGS, minimizer, Fminbox


stable_distance(a, b) = sqrt(sum(abs2, a .- b) + 1e-12)

function tdoa_mle(xs, tdoa_set, speed_of_sound; y_init=[0.0, 0.0, 0.0])

    Q = pairwise_difference_matrix(length(xs))

    fixlogz(y) = [y[1], y[2], exp(y[3])]

    f(θ) = begin
        y = fixlogz(θ)
        delay = map(xs) do x
            d = stable_distance(x, y)
            return d / speed_of_sound
        end
        tdoa_estimated = Q * delay
        sum((tdoa_estimated .- tdoa_set) .^ 2)
    end

    result = optimize(
        f,
        y_init,
        BFGS()
    )

    return fixlogz(result.minimizer), result.minimum
end

function tdoa_mle_2d(xs, tdoa_measured, speed_of_sound; y_init=[0.0, 0.0])
    Q = pairwise_difference_matrix(length(xs))
    fixlogy(y) = [y[1], exp(y[2])]

    f(θ) = begin
        y = fixlogy(θ)
        delay = map(xs) do x
            d = stable_distance(x, y)
            return d / speed_of_sound
        end
        tdoa_estimated = Q * delay
        sum((tdoa_estimated .- tdoa_measured) .^ 2)
    end

    result = optimize(
        f,
        y_init,
        BFGS(); # quasi-Newton
    )

    return fixlogy(result.minimizer), result.minimum
end



function toa_mle(xs, toa_set, speed_of_sound; y_init=[0.0, 0.0, 0.0, 0.0])
    fixlogz(y) = [y[1], y[2], exp(y[3])]

    f(θ) = begin
        y = fixlogz(θ[1:3])
        t₀ = θ[4]
        
        delay = map(xs) do x
            d = stable_distance(x, y)
            return d / speed_of_sound
        end
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

function toa_mle_2d(xs, toa_measured, speed_of_sound; y_init=[0.0, 0.0, 0.0])

    fixlogy(y) = [y[1], exp(y[2])]

    f(θ) = begin
        y = fixlogy(θ[1:2])
        t₀ = θ[3]
        delay = map(xs) do x
            d = stable_distance(x, y)
            return d / speed_of_sound
        end
        toa_estimated = delay .+ t₀
        sum((toa_estimated .- toa_measured) .^ 2)
    end

    result = optimize(
        f,
        y_init,
        BFGS(); # quasi-Newton
    )

    return fixlogy(result.minimizer[1:2]), result.minimizer[3], result.minimum
end




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
                d = stable_distance(x, [y_xs[i], y_ys[i]])
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