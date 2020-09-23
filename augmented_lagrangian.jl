using Plots, LinearAlgebra
include(joinpath(pwd(),"src/moi.jl"))
include(joinpath(pwd(),"src/moi_augmented_lagrangian.jl"))

# MOI setup
"""
 min x'Px
 st Ax <= b
    C*x = d
"""

n = 100
mc = 20
m = 2*mc
P = Diagonal(rand(n))
A = rand(mc,n)
b = rand(mc)
C = rand(mc,n)
d = rand(mc)

function obj(z)
    z'*P*z
end

function con!(c,z)
    c[1:mc] = A*z - b
    c[mc .+ (1:mc)] = C*z - d
    return c
end

function con_ineq!(c,z)
    c .= A*z - b
    return c
end

function con_eq!(c,z)
    c .= C*z - d
    return c
end

prob = ProblemMOI(n,2*mc,idx_ineq=(1:mc))
al_prob = ALProblem(n,mc,mc)

z0 = rand(n)
@time z_sol = solve(copy(z0),prob)
@time z_sol = solve(copy(z0),al_prob)#,c_tol=1.0e-4,tol=1.0e-4)

con_eq!(al_prob.c_eq,z_sol)
con_ineq!(al_prob.c_ineq,z_sol)
norm(al_prob.c_eq,Inf)
norm(max.(0.0,al_prob.c_ineq),Inf)
