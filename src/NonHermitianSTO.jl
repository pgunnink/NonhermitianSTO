module NonHermitianSTO
using DifferentialEquations
using Random, Distributions
using Parameters
using DrWatson
using StaticArrays
import SpecialFunctions.expint

export Settings, Parameters, ode, constant_parameters, init_u0, random_ω
export f_full!, f_quadratic!, f_dipole!
export constant_power, init_u0, init_u0_T, p0
export dict, determine_f
export init_c0
export ParamSTO, Params
export array_complex, stochastic_array, g_STO, g_system

export dipole_system_complex
export constant_power, complete_system_G
export ham

export run_and_save


include("definitions.jl")
include("stochastic.jl")
include("systems.jl")
include("run.jl")
function init_u0(λ, N)
    u = zeros(Float64, 4N)
    init_ϕ = rand(Float64, 2N) * 2π
    dist = Exponential(λ / 2)
    init_p = rand(dist, 2N)
    u[1:2:4N] = init_p
    u[2:2:4N] = init_ϕ
    u
end

function init_c0(p, N)
    c0 = zeros(ComplexF64, 2N)
    for i in 1:N
        dist = Exponential(p.A[i].D / 2)
        c0[2 * (i - 1) + 1] = sqrt(rand(dist)) * exp(im * rand() * 2pi)
        dist = Exponential(p.B[i].D / 2)
        c0[2 * (i - 1) + 2] = sqrt(rand(dist)) * exp(im * rand() * 2pi)
    end
    c0
end

function init_c0(p::ParamSTO)
    dist = Exponential(p.D / 2)
    sqrt(rand(dist)) * exp(im * rand() * 2pi)
end

function ham(s::Settings)
    N = s.N
    H = zeros(ComplexF64, 2N, 2N)
    
    for i in 1:N
        iA = (i - 1) * 2 + 1
        iB = (i - 1) * 2 + 2
        # intra cell coupling:
        H[iA,iA] = s.ω_A + im * (s.J_s_A - s.ω_A * s.α_A - s.ω_A * (s.G_tilde + s.G)) 
        H[iB,iB] = s.ω_B + im * (s.J_s_B - s.ω_B * s.α_B - s.ω_B * (s.G_tilde + s.G)) 
        H[iA,iB] = -s.J + im * s.G_tilde * s.ω_A
        H[iB,iA] = -s.J + im * s.G_tilde * s.ω_A

        if i > 1
            H[iA, iB - 2] = -s.J_tilde + im * s.G_tilde * s.ω_A
            H[iB - 2, iA] = -s.J_tilde + im * s.G_tilde * s.ω_A
        end
        if i < N
            H[iB, iA + 2] = -s.J_tilde + im * s.G_tilde * s.ω_A
            H[iA + 2, iB] = -s.J_tilde + im * s.G_tilde * s.ω_A
        end
    end
    return H
end


function p0(ω, α, Js)
    ζ = Js / (α * ω)
    Q = 2 / ω - 1
    (ζ - 1) / (ζ + Q)
end
function p0(s::Settings)
    p0(s.ω_A, s.α_A, s.J_s_A)
end

function p0(p::ParamSTO)
    p0(p.ω, p.α, p.Js)

end



function f_full!(du, u, pp, t)
    # unpack parameters and temporary variables
    p, c_A, c_B = pp
    # p_A   p_B     ϕ_A     ϕ_B
    # 1:N   N+1:2N  2N+1:3N 3N+1:4N
    N = length(p.ω_A)
    p_A = @view u[1:4:4N]
    p_B = @view u[3:4:4N]
    for x in [p_A, p_B]
        idx = findall(y -> y < 0, x)
        x[idx] .= eps(Float64)
    end

    ϕ_A = @view u[2:4:4N]
    ϕ_B = @view u[4:4:4N]
    c_A = sqrt.(p_A .* (1 .- p_A))
    c_B = sqrt.(p_B .* (1 .- p_B))
    @inbounds for i = 1:N
        # REMEMBER: du[4(i-1)+1] = p_A[i]
        # single STO:
        du[4 * (i - 1) + 1] = 2p_A[i] * (p_A[i] - 1) * (p.α_A[i] * p.ω_A[i] - p.A[i].Js + 2p_A[i] * p.α_A[i])

        # G_i m_Ai × ∂m_Ai:
        du[4 * (i - 1) + 1] += 2p.G[i] * p_A[i] * (p_A[i] - 1) * (p.ω_A[i] + 2p_A[i])
        if i > 1
            du[4 * (i - 1) + 1] += 2p.J_tilde[i - 1] * c_A[i] * c_B[i - 1] * sin(ϕ_A[i] - ϕ_B[i - 1])
            # -G_tilde_i-1 m_Bi × ∂m_Bi:
            du[4 * (i - 1) + 1] -= 2p.G_tilde[i - 1] * p_B[i] * (p_B[i] - 1) * (p.ω_B[i] + 2p_B[i])
            # G_tilde_i-1 m_Ai × ∂m_Ai:
            du[4 * (i - 1) + 1] += 2p.G_tilde[i - 1] * p_A[i] * (p_A[i] - 1) * (p.ω_A[i] + 2p_A[i])
            # -G_tilde_i-1 m_Bi-1 × ∂m_Bi-1:
            du[4 * (i - 1) + 1] -= 2p.G_tilde[i - 1] * p_B[i - 1] * (p_B[i - 1] - 1) * (p.ω_B[i - 1] + 2p_B[i - 1])
        end
        du[4 * (i - 1) + 1] += 2p.J[i] * c_A[i] * c_B[i] * sin(ϕ_A[i] - ϕ_B[i])

        # REMEMBER: du[4*(i-1)+3] = p_B[i]
        du[4 * (i - 1) + 3] = 2p_B[i] * (p_B[i] - 1) * (p.B[i].α * p.ω_B[i] - p.J_s_B[i] + 2p_B[i] * p.B[i].α)
       
        # G_i m_Bi × ∂m_Bi + G_tilde_i m_Bi × ∂m_Bi
        du[4 * (i - 1) + 3] += 2 * (p.G[i] + p.G_tilde[i]) * p_B[i] * (p_B[i] - 1) * (p.ω_B[i] + 2p_B[i])
        if i < N
            du[4 * (i - 1) + 3] += 2p.J_tilde[i] * c_A[i + 1] * c_B[i] * sin(ϕ_B[i] - ϕ_A[i + 1])
            # -G_tilde_i-1 m_Ai+1 × ∂m_Ai+1:
            du[4 * (i - 1) + 1] -= 2p.G_tilde[i] * p_A[i + 1] * (p_A[i + 1] - 1) * (p.ω_A[i + 1] + 2p_A[i + 1])
        end
        # -G_i-1 m_Ai × ∂m_Ai:
        du[4 * (i - 1) + 1] -= 2p.G[i] * p_A[i] * (p_A[i] - 1) * (p.ω_A[i] + 2p_A[i])
        du[4 * (i - 1) + 3] += 2p.J[i] * c_A[i] * c_B[i] * sin(ϕ_B[i] - ϕ_A[i])


        # REMEMBER: du[4*(i-1)+2] = ϕ_A[i]:
        du[4 * (i - 1) + 2] = -p.ω_A[i] - 2p_A[i]
        # -G_i-1 m_Bi × ∂m_Bi:
        du[4 * (i - 1) + 2] -= p.G[i] * c_A[i] / c_B[i] * (2p_B[i] - 1) * 
            (p.ω_B[i] + 2p_B[i]) * sin(ϕ_B[i] - ϕ_A[i])

        du[4 * (i - 1) + 2] += p.J[i] * (2p_B[i] - 1 - c_B[i] * (2p_A[i] - 1) * 
            cos(ϕ_A[i] - ϕ_B[i]) / c_A[i]  )
        if i > 1
            du[4 * (i - 1) + 2] += p.J_tilde[i - 1] * 
                (2p_B[i - 1] - 1 - c_B[i - 1] / c_A[i] * (2p_A[i] - 1) * 
                cos(ϕ_A[i] - ϕ_B[i - 1]))
            # -G_tilde_i-1 m_Bi-1 × ∂m_Bi-1:
            du[4 * (i - 1) + 2] -= p.G_tilde[i - 1] * c_A[i] / c_B[i - 1] * 
                (2p_B[i - 1] - 1) * (p.ω_B[i - 1] + 2p_B[i - 1]) * sin(ϕ_B[i - 1] - ϕ_A[i])

        end

           

        # REMEMBER: du[4*(i-1)+4] = ϕ_B[i]:
        du[4 * (i - 1) + 4] = -p.ω_B[i] - 2p_B[i]
        # -G_i-1 m_Ai × ∂m_Ai:
        du[4 * (i - 1) + 4] -= p.G[i] * c_B[i] / c_A[i] * (2p_A[i] - 1) * 
            (p.ω_A[i] + 2p_A[i]) * sin(ϕ_A[i] - ϕ_B[i])
        du[4 * (i - 1) + 4] += p.J[i] * (2p_A[i] - 1 - c_A[i]  * (2p_B[i] - 1) * 
            cos(ϕ_B[i] - ϕ_A[i]) / c_B[i])
        if i < N
            du[4 * (i - 1) + 4] += p.J_tilde[i] * 
                (2p_A[i + 1] - 1 - c_A[i + 1] / c_B[i] * (2p_B[i] - 1) * 
                    cos(ϕ_B[i] - ϕ_A[i + 1]))
            
            # -G_tilde_i m_Ai+1 × ∂m_Ai+1:
            du[4 * (i - 1) + 4] -= p.G_tilde[i] * c_B[i] / c_A[i + 1] * 
                (2p_A[i + 1] - 1) * (p.ω_A[i + 1] + 2p_A[i + 1]) * sin(ϕ_A[i + 1] - ϕ_B[i])
        end
    end
end

function f_quadratic!(du, u, pp, t)
    # unpack parameters and temporary variables
    p, c_A, c_B = pp
    # p_A   p_B     ϕ_A     ϕ_B
    # 1:N   N+1:2N  2N+1:3N 3N+1:4N
    N = length(p.ω_A)
    p_A = @view u[1:4:4N]
    p_B = @view u[3:4:4N]
    for x in [p_A, p_B]
        idx = findall(y -> y < 0, x)
        x[idx] .= eps(Float64)
    end

    ϕ_A = @view u[2:4:4N]
    ϕ_B = @view u[4:4:4N]
    c_A = sqrt.(p_A .* (1 .- p_A))
    c_B = sqrt.(p_B .* (1 .- p_B))
    @inbounds for i = 1:N
        # REMEMBER: du[4(i-1)+1] = p_A[i]
        # single STO:
        # du[4 * (i - 1) + 1] = 2p_A[i] * (p_A[i] - 1) * (p.α_A[i] * p.ω_A[i] - p.J_s_A[i] + 2p_A[i] * p.α_A[i])
        # only up to 2nd order:
        du[4 * (i - 1) + 1] = 2p_A[i] * (p_A[i] - 1) * (p.α_A[i] * p.ω_A[i] - p.J_s_A[i])
        du[4 * (i - 1) + 1] -= 2p_A[i] * 2p_A[i] * p.α_A[i]
        # G_i m_Ai × ∂m_Ai:
        du[4 * (i - 1) + 1] += 2p.G[i] * p_A[i] * (p_A[i] - 1) * (p.ω_A[i] + 2p_A[i])
        if i > 1
            du[4 * (i - 1) + 1] += 2p.J_tilde[i - 1] * c_A[i] * c_B[i - 1] * sin(ϕ_A[i] - ϕ_B[i - 1])
            # -G_tilde_i-1 m_Bi × ∂m_Bi:
            du[4 * (i - 1) + 1] -= 2p.G_tilde[i - 1] * p_B[i] * (p_B[i] - 1) * (p.ω_B[i] + 2p_B[i])
            # G_tilde_i-1 m_Ai × ∂m_Ai:
            du[4 * (i - 1) + 1] += 2p.G_tilde[i - 1] * p_A[i] * (p_A[i] - 1) * (p.ω_A[i] + 2p_A[i])
            # -G_tilde_i-1 m_Bi-1 × ∂m_Bi-1:
            du[4 * (i - 1) + 1] -= 2p.G_tilde[i - 1] * p_B[i - 1] * (p_B[i - 1] - 1) * (p.ω_B[i - 1] + 2p_B[i - 1])
        end
        du[4 * (i - 1) + 1] += 2p.J[i] * c_A[i] * c_B[i] * sin(ϕ_A[i] - ϕ_B[i])

        # REMEMBER: du[4*(i-1)+3] = p_B[i]
        # du[4 * (i - 1) + 3] = 2p_B[i] * (p_B[i] - 1) * (p.α_B[i] * p.ω_B[i] - p.J_s_B[i] + 2p_B[i] * p.α_B[i])
        # only up to 2nd order:
        du[4 * (i - 1) + 3] = 2p_B[i] * (p_B[i] - 1) * (p.α_B[i] * p.ω_B[i] - p.J_s_B[i])
        du[4 * (i - 1) + 3] -= 2p_B[i] * 2p_B[i] * p.α_B[i]
        # G_i m_Bi × ∂m_Bi + G_tilde_i m_Bi × ∂m_Bi
        du[4 * (i - 1) + 3] += 2 * (p.G[i] + p.G_tilde[i]) * p_B[i] * (p_B[i] - 1) * (p.ω_B[i] + 2p_B[i])
        if i < N
            du[4 * (i - 1) + 3] += 2p.J_tilde[i] * c_A[i + 1] * c_B[i] * sin(ϕ_B[i] - ϕ_A[i + 1])
            # -G_tilde_i-1 m_Ai+1 × ∂m_Ai+1:
            du[4 * (i - 1) + 1] -= 2p.G_tilde[i] * p_A[i + 1] * (p_A[i + 1] - 1) * (p.ω_A[i + 1] + 2p_A[i + 1])
        end
        # -G_i-1 m_Ai × ∂m_Ai:
        du[4 * (i - 1) + 1] -= 2p.G[i] * p_A[i] * (p_A[i] - 1) * (p.ω_A[i] + 2p_A[i])
        du[4 * (i - 1) + 3] += 2p.J[i] * c_A[i] * c_B[i] * sin(ϕ_B[i] - ϕ_A[i])


        # REMEMBER: du[4*(i-1)+2] = ϕ_A[i]:
        du[4 * (i - 1) + 2] = -p.ω_A[i] - 2p_A[i]
        # -G_i-1 m_Bi × ∂m_Bi:
        du[4 * (i - 1) + 2] -= p.G[i] * c_A[i] / c_B[i] * (2p_B[i] - 1) * 
            (p.ω_B[i] + 2p_B[i]) * sin(ϕ_B[i] - ϕ_A[i])

        du[4 * (i - 1) + 2] += p.J[i] * (2p_B[i] - 1 - c_B[i] * (2p_A[i] - 1) * 
            cos(ϕ_A[i] - ϕ_B[i]) / c_A[i]  )
        if i > 1
            du[4 * (i - 1) + 2] += p.J_tilde[i - 1] * 
                (2p_B[i - 1] - 1 - c_B[i - 1] / c_A[i] * (2p_A[i] - 1) * 
                cos(ϕ_A[i] - ϕ_B[i - 1]))
            # -G_tilde_i-1 m_Bi-1 × ∂m_Bi-1:
            du[4 * (i - 1) + 2] -= p.G_tilde[i - 1] * c_A[i] / c_B[i - 1] * 
                (2p_B[i - 1] - 1) * (p.ω_B[i - 1] + 2p_B[i - 1]) * sin(ϕ_B[i - 1] - ϕ_A[i])

        end

           

        # REMEMBER: du[4*(i-1)+4] = ϕ_B[i]:
        du[4 * (i - 1) + 4] = -p.ω_B[i] - 2p_B[i]
        # -G_i-1 m_Ai × ∂m_Ai:
        du[4 * (i - 1) + 4] -= p.G[i] * c_B[i] / c_A[i] * (2p_A[i] - 1) * 
            (p.ω_A[i] + 2p_A[i]) * sin(ϕ_A[i] - ϕ_B[i])
        du[4 * (i - 1) + 4] += p.J[i] * (2p_A[i] - 1 - c_A[i]  * (2p_B[i] - 1) * 
            cos(ϕ_B[i] - ϕ_A[i]) / c_B[i])
        if i < N
            du[4 * (i - 1) + 4] += p.J_tilde[i] * 
                (2p_A[i + 1] - 1 - c_A[i + 1] / c_B[i] * (2p_B[i] - 1) * 
                    cos(ϕ_B[i] - ϕ_A[i + 1]))
            
            # -G_tilde_i m_Ai+1 × ∂m_Ai+1:
            du[4 * (i - 1) + 4] -= p.G_tilde[i] * c_B[i] / c_A[i + 1] * 
                (2p_A[i + 1] - 1) * (p.ω_A[i + 1] + 2p_A[i + 1]) * sin(ϕ_A[i + 1] - ϕ_B[i])
        end

        
    end
end



function f_dipole!(du, u, pp, t)
    # unpack parameters and temporary variables
    p, c_A, c_B = pp
    # p_A   p_B     ϕ_A     ϕ_B
    # 1:N   N+1:2N  2N+1:3N 3N+1:4N
    N = length(p.ω_A)
    p_A = @view u[1:4:4N]
    p_B = @view u[3:4:4N]
    for x in [p_A, p_B]
        idx = findall(y -> y < 0, x)
        x[idx] .= eps(Float64)
    end
    
    ϕ_A = @view u[2:4:4N]
    ϕ_B = @view u[4:4:4N]

    # if (any(x -> x > 1, p_A)) || (any(x -> x > 1, p_B))
    #     @info p_A
    #     @info ϕ_A
    #     @info p_B
    #     @info ϕ_B
    # end
    c_A = sqrt.(p_A .* (1 .- p_A))
    c_B = sqrt.(p_B .* (1 .- p_B))
    @inbounds for i = 1:N
        # REMEMBER: du[4(i-1)+1] = p_A[i]
        # single STO:
        # du[4 * (i - 1) + 1] = 2p_A[i] * (p_A[i] - 1) * (p.α_A[i] * p.ω_A[i] - p.J_s_A[i] + 2p_A[i] * p.α_A[i])
        
        # only up to 2nd order:
        du[4 * (i - 1) + 1] = 2p_A[i] * (p_A[i] - 1) * (p.A[i].α * p.A[i].ω - p.A[i].Js)
        du[4 * (i - 1) + 1] -= 2p_A[i] * 2p_A[i] * p.A[i].α
        
        # G_i m_Ai × ∂m_Ai:
        du[4 * (i - 1) + 1] += 2p.G[i] * p_A[i] * (p_A[i] - 1) * (p.A[i].ω + 2p_A[i])
        if i > 1
            du[4 * (i - 1) + 1] += 2p.J_tilde[i - 1] * c_A[i] * c_B[i - 1] * sin(ϕ_A[i] - ϕ_B[i - 1])
            # -G_tilde_i-1 m_Bi × ∂m_Bi:
            du[4 * (i - 1) + 1] -= 2p.G_tilde[i - 1] * p_B[i] * (p_B[i] - 1) * (p.ω_B[i] + 2p_B[i])
            # G_tilde_i-1 m_Ai × ∂m_Ai:
            du[4 * (i - 1) + 1] += 2p.G_tilde[i - 1] * p_A[i] * (p_A[i] - 1) * (p.A[i].ω + 2p_A[i])
            # -G_tilde_i-1 m_Bi-1 × ∂m_Bi-1:
            du[4 * (i - 1) + 1] -= 2p.G_tilde[i - 1] * p_B[i - 1] * (p_B[i - 1] - 1) * (p.B[i - 1].ω + 2p_B[i - 1])
        end
        du[4 * (i - 1) + 1] += 2p.J[i] * c_A[i] * c_B[i] * sin(ϕ_A[i] - ϕ_B[i])

        # REMEMBER: du[4*(i-1)+3] = p_B[i]
        # du[4 * (i - 1) + 3] = 2p_B[i] * (p_B[i] - 1) * (p.α_B[i] * p.ω_B[i] - p.J_s_B[i] + 2p_B[i] * p.α_B[i])
        
        # only up to 2nd order:
        du[4 * (i - 1) + 3] = 2p_B[i] * (p_B[i] - 1) * (p.B[i].α * p.B[i].ω - p.B[i].Js)
        du[4 * (i - 1) + 3] -= 2p_B[i] * 2p_B[i] * p.B[i].α
       
        # G_i m_Bi × ∂m_Bi + G_tilde_i m_Bi × ∂m_Bi
        du[4 * (i - 1) + 3] += 2 * (p.G[i] + p.G_tilde[i]) * p_B[i] * (p_B[i] - 1) * (p.B[i].ω + 2p_B[i])
        if i < N
            du[4 * (i - 1) + 3] += 2p.J_tilde[i] * c_A[i + 1] * c_B[i] * sin(ϕ_B[i] - ϕ_A[i + 1])
            # -G_tilde_i-1 m_Ai+1 × ∂m_Ai+1:
            du[4 * (i - 1) + 1] -= 2p.G_tilde[i] * p_A[i + 1] * (p_A[i + 1] - 1) * (p.A[i + 1].ω + 2p_A[i + 1])
        end
        # -G_i-1 m_Ai × ∂m_Ai:
        du[4 * (i - 1) + 1] -= 2p.G[i] * p_A[i] * (p_A[i] - 1) * (p.A[i].ω + 2p_A[i])
        du[4 * (i - 1) + 3] += 2p.J[i] * c_A[i] * c_B[i] * sin(ϕ_B[i] - ϕ_A[i])


        # REMEMBER: du[4*(i-1)+2] = ϕ_A[i]:
        du[4 * (i - 1) + 2] = -p.A[i].ω - 2p_A[i]
        # -G_i-1 m_Bi × ∂m_Bi:
        du[4 * (i - 1) + 2] -= p.G[i] * c_A[i] / c_B[i] * (2p_B[i] - 1) * 
            (p.B[i].ω + 2p_B[i]) * sin(ϕ_B[i] - ϕ_A[i])

        du[4 * (i - 1) + 2] += p.J[i] * (2p_B[i] - 1 - c_B[i] * (2p_A[i] - 1) * 
            cos(ϕ_A[i] - ϕ_B[i]) / c_A[i]  )
        if i > 1
            du[4 * (i - 1) + 2] += p.J_tilde[i - 1] * 
                (2p_B[i - 1] - 1 - c_B[i - 1] / c_A[i] * (2p_A[i] - 1) * 
                cos(ϕ_A[i] - ϕ_B[i - 1]))
            # -G_tilde_i-1 m_Bi-1 × ∂m_Bi-1:
            du[4 * (i - 1) + 2] -= p.G_tilde[i - 1] * c_A[i] / c_B[i - 1] * 
                (2p_B[i - 1] - 1) * (p.B[i - 1].ω + 2p_B[i - 1]) * sin(ϕ_B[i - 1] - ϕ_A[i])

        end

           

        # REMEMBER: du[4*(i-1)+4] = ϕ_B[i]:
        du[4 * (i - 1) + 4] = -p.B[i].ω - 2p_B[i]
        # -G_i-1 m_Ai × ∂m_Ai:
        du[4 * (i - 1) + 4] -= p.G[i] * c_B[i] / c_A[i] * (2p_A[i] - 1) * 
            (p.A[i].ω + 2p_A[i]) * sin(ϕ_A[i] - ϕ_B[i])
        du[4 * (i - 1) + 4] += p.J[i] * (2p_A[i] - 1 - c_A[i]  * (2p_B[i] - 1) * 
            cos(ϕ_B[i] - ϕ_A[i]) / c_B[i])
        if i < N
            du[4 * (i - 1) + 4] += p.J_tilde[i] * 
                (2p_A[i + 1] - 1 - c_A[i + 1] / c_B[i] * (2p_B[i] - 1) * 
                    cos(ϕ_B[i] - ϕ_A[i + 1]))
            
            # -G_tilde_i m_Ai+1 × ∂m_Ai+1:
            du[4 * (i - 1) + 4] -= p.G_tilde[i] * c_B[i] / c_A[i + 1] * 
                (2p_A[i + 1] - 1) * (p.ω_A[i + 1] + 2p_A[i + 1]) * sin(ϕ_A[i + 1] - ϕ_B[i])
        end


        # dipolar interactions
        @inbounds for j in 1:N
            if i != j # exclude self interaction
                # A-A interactions:
                # p_A[i]:
                du[4 * (i - 1) + 1] += p.dip_A[i] / (abs(2i - 2j))^3 * 
                    (
                    -2 * c_A[i] * c_A[j] * 
                        (
                            cos(ϕ_A[j]) * sin(ϕ_A[i]) +
                            2cos(ϕ_A[i]) * sin(ϕ_A[j])
                        )
                    )
                # ϕ_A[i]
                du[4 * (i - 1) + 2] += p.dip_A[i] / (abs(2i - 2j))^3 * 
                (
                    1 - 2p_A[j] + c_A[j] / c_A[i] * (-1 + 2p_A[i]) * 
                    (
                        cos(ϕ_A[i]) * cos(ϕ_A[j]) -
                        2sin(ϕ_A[i]) * sin(ϕ_A[j])
                    )
                )
                # B-B interactions:
                # p_B[i]:
                du[4 * (i - 1) + 3] += p.dip_B[i] / (abs(2i - 2j))^3 * 
                    (
                    -2 * c_B[i] * c_B[j] * 
                        (
                            cos(ϕ_B[j]) * sin(ϕ_B[i]) +
                            2cos(ϕ_B[i]) * sin(ϕ_B[j])
                        )
                    )
                # ϕ_B[i]
                du[4 * (i - 1) + 4] += p.dip_B[i] / (abs(2i - 2j))^3 * 
                (
                    1 - 2p_B[j] + c_B[j] / c_B[i] * (-1 + 2p_B[i]) * 
                    (
                        cos(ϕ_B[i]) * cos(ϕ_B[j]) -
                        2sin(ϕ_B[i]) * sin(ϕ_B[j])
                    )
                )
            end
            # A(i) - B(j) interactions
            # p_A[i]
            du[4 * (i - 1) + 1] += p.dip_A[i] / (abs(2i - 2j - 1))^3 * 
                (
                -2 * c_A[i] * c_B[j] * 
                    (
                        cos(ϕ_B[j]) * sin(ϕ_A[i]) +
                        2cos(ϕ_A[i]) * sin(ϕ_B[j])
                    )
                )
            # ϕ_A[i]
            du[4 * (i - 1) + 2] += p.dip_A[i] / (abs(2i - 2j - 1))^3 * 
                (
                    1 - 2p_B[j] + c_B[j] / c_A[i] * (-1 + 2p_A[i]) * 
                    (
                        cos(ϕ_A[i]) * cos(ϕ_B[j]) -
                        2sin(ϕ_A[i]) * sin(ϕ_B[j])
                    )
                )
            # B(i) - A(j) interactions
            # p_B[i]
            du[4 * (i - 1) + 3] += p.dip_B[i] / (abs(2i - 2j + 1))^3 * 
                (
                -2 * c_B[i] * c_B[j] * 
                    (
                        cos(ϕ_B[j]) * sin(ϕ_B[i]) +
                        2cos(ϕ_B[i]) * sin(ϕ_B[j])
                    )
                )
            # ϕ_B[i]
            du[4 * (i - 1) + 4] += p.dip_B[i] / (abs(2i - 2j + 1))^3 * 
                (
                    1 - 2p_B[j] + c_B[j] / c_B[i] * (-1 + 2p_B[i]) * 
                    (
                        cos(ϕ_B[i]) * cos(ϕ_B[j]) -
                        2sin(ϕ_B[i]) * sin(ϕ_B[j])
                    )
                )
        end
    end
end

end # module
