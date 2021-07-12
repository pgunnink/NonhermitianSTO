
function ω_p(power, ω)
    ω + 2power
end

function Γplus(power, ω, α)
    α * ω * (1 + (2 / ω - 1) * power)
end

function Γmin(power, Js)
    Js * (1 - power)
end


function single_sto(c, ω, α, Js)
    power = abs(c)^2
    -1im * (ω + 2power) * c - 
        α * ω * (1  + (2 / ω - 1) * power) * c +
        Js * (1 - power) * c
end

function single_sto_linear(c, p::ParamSTO)
    -1im * p.ω * c - 
        p.ω * p.α * c +
        p.Js * c
end

function single_sto(c, p::ParamSTO)
    power = abs(c)^2
    -1im * ω_p(power, p.ω) * c - 
        Γplus(power, p.ω, p.α) * c +
        Γmin(power, p.Js) * c
end

function coupling_G(cA, cB, ωA)
    # coupling ∂ₜcB = cA × ∂ₜcA
    pA = abs(cA)^2
    pB = abs(cB)^2
    # res = cA * (ωA + 2pA) * (-√(1 - pA) * √(1 - pB) + (-cB + cB * pA + 2cA * √(1 - pA) * √(1 - pB)) * conj(cA))

    res = cA * ωA * (-√(1 - pA) * √(1 - pB) + (-cB + 2cA * √(1 - pA) * √(1 - pB)) * conj(cA))
    res += cA * 2pA * (-√(1 - pA) * √(1 - pB) - cB  * conj(cA))

    return -res / (pB - 1)
end


function array_g_complex(c, p, t)
    N = length(p.A)
    cA = @view c[1:2:2N]
    cB = @view c[2:2:2N]
    
    res = zeros(ComplexF64, 2N)
    for i in 1:N 
        # G_i m_Ai × ∂m_Ai:
        # res[2 * (i - 1) + 1] = -Γplus(abs(cA[i])^2, p.A[i].ω, p.G[i]) * cA[i]
        # -G_i m_Bi × ∂m_Bi:
        res[2 * (i - 1) + 1] -= p.G[i] * coupling_G(cB[i], cA[i], p.B[i].ω)
        if i > 1
            # G_tilde_i-1 m_Ai × ∂m_Ai:
            # res[2 * (i - 1) + 1] += -Γplus(abs(cA[i])^2, p.A[i].ω, p.G_tilde[i - 1]) * cA[i]
            # -G_tilde_i-1 m_Bi-1 × ∂m_Bi-1:
            res[2 * (i - 1) + 1] -= p.G_tilde[i - 1] * coupling_G(cB[i - 1], cA[i], p.B[i - 1].ω)
        end

        # G_i m_Bi × ∂m_Bi + G_tilde_i m_Bi × ∂m_Bi
        # res[2 * (i - 1) + 2] = -Γplus(abs(cB[i])^2, p.B[i].ω, p.G[i] + p.G_tilde[i]) * cB[i]
        # -G_i m_Ai × ∂m_Ai:
        res[2 * (i - 1) + 2] -= p.G[i] * coupling_G(cA[i], cB[i], p.A[i].ω)
        if i < N
            # -G_tilde_i m_Ai+1 × ∂m_Ai+1:
            res[2 * (i - 1) + 2] -= p.G_tilde[i] * coupling_G(cA[i + 1], cB[i], p.A[i + 1].ω)
        end
    end
    res
end

function coupling_J(cA, cB)
    pA = abs(cA)^2
    pB = abs(cB)^2
    im / (2sqrt(1 - pA)) * 
            (
                -2cA * sqrt(1 - pA) + 
                cB * (2 - 3pA) * sqrt(1 - pB) + 
                cA * conj(cB) * (4cB * sqrt(1 - pA) - cA * sqrt(1 - pB))
                )
end
function array_complex(c, p, t)
    # c is layout as: cA1 cB1 cA1 cB1 etc...
    N = length(p.A)
    cA = @view c[1:2:2N]
    cB = @view c[2:2:2N]
    pA = abs.(cA).^2
    pB = abs.(cB).^2
    res = zeros(ComplexF64, 2N)
    for i in 1:N 
        res[2 * (i - 1) + 1] = single_sto(cA[i], p.A[i])

        # J coupling:
        res[2 * (i - 1) + 1] += im * p.J[i] / (2sqrt(1 - pA[i])) * 
            (
                -2cA[i] * sqrt(1 - pA[i]) + 
                cB[i] * (2 - 3pA[i]) * sqrt(1 - pB[i]) + 
                cA[i] * conj(cB[i]) * (4cB[i] * sqrt(1 - pA[i]) - cA[i] * sqrt(1 - pB[i]))
                )
        if i > 1
            # J_tilde coupling
            res[2 * (i - 1) + 1] += im * p.J_tilde[i - 1] / (2sqrt(1 - pA[i])) * 
                (
                    -2cA[i] * sqrt(1 - pA[i]) + 
                    cB[i - 1] * (2 - 3pA[i]) * sqrt(1 - pB[i - 1]) + 
                    cA[i] * conj(cB[i - 1]) * (4cB[i - 1] * sqrt(1 - pA[i]) - cA[i] * sqrt(1 - pB[i]))
                )
        end

        res[2 * (i - 1) + 2] = single_sto(cB[i], p.B[i])
        res[2 * (i - 1) + 2] += im * p.J[i] / (2sqrt(1 - pB[i])) * 
                (
                    -2cB[i] * sqrt(1 - pB[i]) + 
                    cA[i] * (2 - 3pB[i]) * sqrt(1 - pB[i]) + 
                    cB[i] * conj(cA[i]) * (4cA[i] * sqrt(1 - pB[i]) - cB[i] * sqrt(1 - pA[i])) 
                )
        if i < N
            res[2 * (i - 1) + 2] += im * p.J_tilde[i + 1] / (2sqrt(1 - pB[i])) * 
                (
                    -2cB[i] * sqrt(1 - pB[i]) + 
                    cA[i + 1] * (2 - 3pB[i]) * sqrt(1 - pB[i]) + 
                    cB[i] * conj(cA[i + 1]) * (4cA[i + 1] * sqrt(1 - pB[i]) - cB[i] * sqrt(1 - pA[i + 1]))
                )
        end
    end
    res
end

function dip_interaction(ci, Pi, cj, Pj)
    -im / √(1 - Pi) * (
        -ci * √(1 - Pi) + 
        2ci * Pj * √(1 - Pi) - 
        2ci^2 * √(1 - Pj) * conj(cj) - 
        √(1 - Pj) * (-1 +  3ci * real(ci)) * (cj - 3real(cj))
        )
end

function dipole_interaction_system(c, p, t)
    N = length(p.A)
    cA = @view c[1:2:2N]
    cB = @view c[2:2:2N]
    pA = abs.(cA).^2
    pB = abs.(cB).^2
    res = zeros(ComplexF64, 2N)
    for i ∈ 1:N
        iA = 2 * (i - 1) + 1
        iB = 2 * (i - 1) + 2
        for j ∈ 1:N
            if i != j 
                # interactions between all A sites:
                res[iA] += p.A[i].dip / (abs(2i - 2j))^3 * 
                    dip_interaction(cA[i], pA[i], cA[j], pA[j])
                # interactions between all B sites:
                res[iB] += p.B[i].dip / (abs(2i - 2j))^3 * 
                    dip_interaction(cB[i], pB[i], cB[j], pB[j])
            end
            # A(i) - B(j) interactions:
            res[iA] += p.A[i].dip / (abs(2i - 2j - 1))^3 * 
                dip_interaction(cA[i], pA[i], cB[j], pB[j])
            # B(i) - A(j) interactions:
            res[iB] += p.B[i].dip / (abs(2i - 2j + 1))^3 * 
                dip_interaction(cB[i], pB[i], cA[j], pA[j])
        end
    end
    res
end

dipole_system_complex(c,p,t) = array_complex(c, p, t) + dipole_interaction_system(c, p, t)


complete_system_G(c,p,t) = array_complex(c, p, t) + array_g_complex(c, p, t)