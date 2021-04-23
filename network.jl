np = nr * (ns + 3) + 1
p = randn(Float64, np) .* 0.05;
p[1:nr] .+= 0.8;  # w_b
p[nr * (ns + 1) + 1:nr * (ns + 2)] .+= 0.8;  # w_out
p[nr * (ns + 2) + 1:end - 1] .+= 0.1;  # w_b | w_Ea
p[end] = 0.1;  # slope

function p2vec(p)
    slope = p[end] .* 1.e1
    w_b = p[1:nr] .* (slope * 10.0)
    w_b = clamp.(w_b, 0, 50)

    w_out = reshape(p[nr + 1:nr * (ns + 1)], ns, nr)
    @. w_out[1, :] = clamp(w_out[1, :], -1.1, 0.0)
    @. w_out[end, :] = clamp(abs(w_out[end, :]), 0.0, 1.1)

    if p_cutoff > 0.0
        w_out[findall(abs.(w_out) .< p_cutoff)] .= 0.0
    end

    w_out[ns - 1:ns - 1, :] .=
        -sum(w_out[1:ns - 2, :], dims=1) .- sum(w_out[ns:ns, :], dims=1)

    w_in_Ea = abs.(p[nr * (ns + 1) + 1:nr * (ns + 2)] .* (slope * 100.0))
    w_in_Ea = clamp.(w_in_Ea, 0.0, 300.0)

    w_in_b = abs.(p[nr * (ns + 2) + 1:nr * (ns + 3)])

    # w_in_ocen = abs.(p[nr*(ns+3)+1:nr*(ns+4)])
    # w_in_ocen = clamp.(w_in_ocen, 0.0, 1.5)

    # if p_cutoff > 0.0
    #     w_in_ocen[findall(abs.(w_in_ocen) .< p_cutoff)] .= 0.0
    # end

    w_in = vcat(clamp.(-w_out, 0.0, 1.1), w_in_Ea', w_in_b')
    return w_in, w_b, w_out
end

function display_p(p)
    w_in, w_b, w_out = p2vec(p)
    println("\n species (column) reaction (row)")
    println("w_in | Ea | b | n_ocen | lnA | w_out")
    show(stdout, "text/plain", round.(hcat(w_in', w_b, w_out'), digits=2))
    # println("\n w_out")
    # show(stdout, "text/plain", round.(w_out', digits=3))
    println("\n")
end
display_p(p);

function getsampletemp(t, T0, beta)
    T = T0 + beta / 60 * t  # K/min to K/s
    return T
end

const R = -1.0 / 8.314e-3  # universal gas constant, kJ/mol*K
@inbounds function crnn!(du, u, p, t)
    logX = @. log(clamp(u, lb, 2.0))
    T = getsampletemp(t, T0, beta)
    w_in_x = w_in' * vcat(logX, R / T, log(T))
    du .= w_out * (@. exp(w_in_x + w_b))
end

tspan = [0.0, 1.0];
u0 = zeros(ns);
u0[1] = 1.0;
prob = ODEProblem(crnn!, u0, tspan, p, abstol=lb)

condition(u, t, integrator) = u[1] < lb * 5.0
affect!(integrator) = terminate!(integrator)
_cb = DiscreteCallback(condition, affect!)

alg = TRBDF2();
sense = ForwardSensitivity(autojacvec=true)
# sense = BacksolveAdjoint()
# sense = ForwardDiffSensitivity()
function pred_n_ode(p, i_exp, exp_data)
    global T0, beta, ocen
    global w_in, w_b, w_out
    T0, beta, ocen = l_exp_info[i_exp, :]
    w_in, w_b, w_out = p2vec(p)

    ts = @view(exp_data[:, 1])
    tspan = [ts[1], ts[end]]
    sol = solve(
        prob,
        alg,
        tspan=tspan,
        p=p,
        saveat=ts,
        sensalg=sense,
        maxiters=maxiters,
        # callback=_cb,
    )

    if sol.retcode == :Success
        nothing
    else
        @sprintf("solver failed beta: %.0f", beta)
    end
    if length(sol.t) > length(ts)
        return  sol[:, 1:length(ts)]
    else
        return sol
    end
end

function loss_neuralode(p, i_exp)
    exp_data = l_exp_data[i_exp]
    pred = Array(pred_n_ode(p, i_exp, exp_data))
    masslist = sum(clamp.(@view(pred[1:end - 1, :]), 0.0, Inf), dims=1)'
    gaslist = clamp.(@views(pred[end, :]), 0.0, Inf)

    s = sample(1:length(masslist), batchsize)

    loss = mae(masslist[s], @view(exp_data[s, 3])) + mae(gaslist[s], 1 .- @view(exp_data[s, 3]))
    # - sum(clamp.(pred, -Inf, -lb)) .* 1.e-3
    return loss
end
@time loss = loss_neuralode(p, 1)
# using BenchmarkTools
# @benchmark loss = loss_neuralode(p, 1)
# @benchmark grad = ForwardDiff.gradient(x -> loss_neuralode(x, 1), p)
