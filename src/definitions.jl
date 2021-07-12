struct ParamSTO
    ω::Float64
    α::Float64
    Js::Float64
    dip::Float64
    D::Float64
end

struct Params
    A::Vector{ParamSTO}
    B::Vector{ParamSTO}
    J_tilde::Vector{Float64}
    J::Vector{Float64}
    G::Vector{Float64}
    G_tilde::Vector{Float64}
    dt::Float64
end




# Base.@kwdef mutable struct Settings
#     N::Int64 = 10
#     ω_A::Float64 = 0.5
#     α_A::Float64 = 1e-2
#     J_s_A::Float64 = 2ω_A * α_A
#     ω_B::Float64 = 0.5
#     α_B::Float64 = 1e-2
#     J_tilde::Float64 = -.025
#     J::Float64 = 0.5J_tilde
#     J_s_B::Float64 = 0
#     G::Float64 = 0
#     G_tilde::Float64 = 0.
#     dip_A::Float64 = 0.
#     dip_B::Float64 = 0.
#     D_A::Float64 = 1e-4 # defined as: kb T / ωM λ = γ kb T / ωM Veff Ms
#     D_B::Float64 = 1e-4 # defined as: kb T / ωM λ = γ kb T / ωM Veff Ms
#     dt::Float64 = 0.1
# end

Base.@kwdef mutable struct Settings
    N::Int64 = 10
    ω_A::Float64 = 0.5
    Δω_A::Float64 = 0
    α_A::Float64 = 1e-2
    Δα_A::Float64 = 0
    J_s_A::Float64 = 2ω_A * α_A
    ΔJ_s_A::Float64 = 0
    ω_B::Float64 = 0.5
    Δω_B::Float64 = 0
    α_B::Float64 = 1e-2
    Δα_B::Float64 = 0
    J_tilde::Float64 = -.025
    ΔJ_tilde::Float64 = 0
    J::Float64 = 0.5J_tilde
    ΔJ::Float64 = 0
    J_s_B::Float64 = 0
    ΔJ_s_B::Float64 = 0
    G::Float64 = 0
    ΔG::Float64 = 0
    G_tilde::Float64 = 0.
    ΔG_tilde::Float64 = 0.
    dip_A::Float64 = 0.
    dip_B::Float64 = 0.
    Δdip_A::Float64 = 0.
    Δdip_B::Float64 = 0.

    D_A::Float64 = 1e-4 # defined as: kb T / ωM λ = γ kb T / ωM Veff Ms
    D_B::Float64 = 1e-4 # defined as: kb T / ωM λ = γ kb T / ωM Veff Ms
    dt::Float64 = 0.1
    
end


function constant_parameters(s::Settings)
    # TODO set G_Tilde, J_Tilde[1,N] == 0
    G_tilde = ones(s.N) * s.G_tilde 
    G_tilde[1] = 0
    G_tilde[end] = 0
    J_tilde = ones(s.N) * s.J_tilde
    # J_tilde[1] = 0
    # J_tilde[end] = 0
    A = [ParamSTO(s.ω_A, s.α_A, s.J_s_A, s.dip_A, s.D_A) for _ in 1:s.N]
    B = [ParamSTO(s.ω_B, s.α_B, s.J_s_B, s.dip_B, s.D_B) for _ in 1:s.N]

    Params(A, B,
        J_tilde,
        ones(s.N) * s.J,
        ones(s.N) * s.G,
        G_tilde, s.dt
        )
end


function random_parameters(s::Settings; fix_Js=false)
    # Δω is variance, 
    G_tilde = ones(s.N) * s.G_tilde + randn(s.N) * √(s.ΔG_tilde)
    G_tilde[1] = 0  
    G_tilde[end] = 0
    J_tilde = ones(s.N) * s.J_tilde + randn(s.N) * √(s.ΔJ_tilde)
    # J_tilde[1] = 0
    # J_tilde[end] = 0

    J = ones(s.N) * s.J + randn(s.N) * √(s.ΔJ)
    G = ones(s.N) * s.G + randn(s.N) * √(s.ΔG)
    α_A = ones(s.N) * s.α_A + randn(s.N) * √(s.Δα_A)
    α_B = ones(s.N) * s.α_B + randn(s.N) * √(s.Δα_B)
    ω_A = ones(s.N) * s.ω_A + randn(s.N) * √(s.Δω_A)
    ω_B = ones(s.N) * s.ω_B + randn(s.N) * √(s.Δω_B)
    if fix_Js
        J_s_A = 2ω_A .* α_A
        
    else
        J_s_A = ones(s.N) * s.J_s_A + randn(s.N) * √(s.ΔJ_s_A)
    end
    J_s_B = ones(s.N) * s.J_s_B + randn(s.N) * √(s.ΔJ_s_B)

    dip_A = ones(s.N) * s.dip_A + randn(s.N) * √(s.Δdip_A)
    dip_B = ones(s.N) * s.dip_B + randn(s.N) * √(s.Δdip_B)

    A = [ParamSTO(ω_A[i], α_A[i] + G[i] + G_tilde[i], J_s_A[i], dip_A[i], s.D_A) for i in 1:s.N]
    B = [ParamSTO(ω_B[i], α_B[i] + G[i] + G_tilde[i], J_s_B[i], dip_B[i], s.D_B) for i in 1:s.N]

    Params(A, B,
        J_tilde,
        J,
        G,
        G_tilde, s.dt
        )
end