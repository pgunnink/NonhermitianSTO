function determine_f(s::Settings)
    function_concat = Function[array_complex]
    if s.G != 0 || s.G_tilde != 0
        push!(function_concat, array_g_complex)
    end
    if ((s.dip_A != 0.) || (s.dip_B != 0.))
        push!(function_concat, dipole_interaction_system)
    end
    new_f(c, p, t) = sum([x(c, p, t) for x in function_concat])
    return new_f
end

function determine_f(p::Params)
    function_concat = Function[array_complex]
    if any(x -> x != 0, p.G) || any(x -> x != 0, p.G_tilde)
        push!(function_concat, array_g_complex)
    end
    if any(x -> x.dip != 0, p.A) || any(x -> x.dip != 0, p.B)
        push!(function_concat, dipole_interaction_system)
    end
    new_f(c, p, t) = sum([x(c, p, t) for x in function_concat])
    return new_f
end

function run_params(p::Params, tend = 1e4)
    N = length(p.A)
    c0 = init_c0(p, N)
    noise = WienerProcess(0., zeros(ComplexF64, 2N))
    f = determine_f(p)
    prob = SDEProblem(f, g_system, c0, (0., tend), p, noise = noise)
    sol = solve(prob, EulerHeun(), dt = p.dt, maxiters = 1e7)
    return sol
end

function run_params_ode(p::Params, tend = 1e4)
    N = length(p.A)
    c0 = init_c0(p, N)
    f = determine_f(p)
    prob = ODEProblem(f,  c0, (0., tend), p)
    sol = solve(prob)
    return sol
end

function get_number_of_lasers(sol, s::Settings)
    y = sol(sol.t[end])[1:4:4s.N]
    return count(x -> x > constant_power(s) * 0.7, y)
end

function run_and_get_number_of_lasers(s::Settings)
    try
        get_number_of_lasers(run_params(s), s)
    catch 
        run_and_get_number_of_lasers(s)
    end
end

function run_N_times(s::Settings, run_N)
    [run_and_get_number_of_lasers(s) for _ in 1:run_N]
end

# convenience function for converting Settings to Dict for easy saving
dict(x::Settings) = Dict{Symbol,Any}(fn => getproperty(x, fn) for fn ∈ propertynames(x))

function run_and_save(s::Settings, save_to; sde = true, tend = 1e4, fix_Js = false)

    p = random_parameters(s; fix_Js = fix_Js)
    

    seed = rand(UInt64)

    Random.seed!(seed)
    
    if sde
        res = run_params(p, tend)
    else
        res = run_params_ode(p, tend)
    end

    to_save = dict(s)
    to_save[:power] = abs.(res[:,end]).^2
    to_save[:seed] = seed
    to_save[:tend] = tend
    if fix_Js
        to_save[:fix_Js] = true
    end
    if !sde
        to_save[:ode] = true
    end
    save(joinpath(save_to, "$seed.bson"), to_save)
end



