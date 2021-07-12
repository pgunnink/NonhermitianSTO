
function p0_T(s::ParamSTO)
    p0_T(s.ω, s.α, s.Js, s.D)
end

function p0_T(ω, α, Js, D)
    Q = 2 / ω - 1
    η = D * ω / 2
    ζ = Js / (α * ω)
    β = -(1 + Q) * ζ / (Q^2 * η)
    integrand = -(ζ + Q) / (Q^2 * η)
    Q * η / (Q + ζ) * (1 + exp(integrand) / expint(β, -integrand)) + (ζ - 1) / (ζ + Q)
end


function g_STO(c, p::ParamSTO, t)
    power = abs(c)^2
    sqrt(p.D * p.ω * Γplus(power, p.ω, p.α) / ω_p(power, p.ω))
    # 0
end

function g_system(c, p, t)
    N = length(p.A)
    res = zeros(ComplexF64, 2N)
    for i in 1:N
        res[2 * (i - 1) + 1] = g_STO(c[2 * (i - 1) + 1], p.A[i], t)
        res[2 * (i - 1) + 2] = g_STO(c[2 * (i - 1) + 2], p.B[i], t)
    end
    res
end