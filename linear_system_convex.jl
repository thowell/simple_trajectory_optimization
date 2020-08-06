using LinearAlgebra, Plots
using Convex, ECOS

"""
min (xT-xF)' QF (xT-xF) + Σ (xt - xF)' Qt (xt - xF) + ut' Rt ut
X,U
s.t xt⁺ = At xt + Bt ut for t = 1:T-1
    x1 = x(t=0)
    xF = x(t=Δt*T)
"""

# planning horizon
T = 10

# double-integrator continuous time
n = 2
m = 1
Ac = [0.0 1.0; 0.0 0.0]
Bc = [0.0; 1.0]

# initial condition
x1 = [1.0; 1.0]

# final condition
xF = [0.0; 0.0]

# double integrator discrete-time dynamics
Δt = 0.1
D = exp(Δt*[Ac Bc; zeros(1,n+m)])
A = [D[1:n,1:n] for t = 1:T-1]
B = [D[1:n,n .+ (1:m)] for t = 1:T-1]

# objective
Q = [Diagonal(ones(n)) for t = 1:T]
R = [Diagonal(1.0e-1*ones(m)) for t = 1:T-1]

# indices for convenience
idx_x = [(t-1)*(n+m) .+ (1:n) for t = 1:T]
idx_u = [(t-1)*(n+m) + n .+ (1:m) for t = 1:T-1]

N = n*T + m*(T-1) # number of decision variables
M = n*(T+1) # number of constraints

# Convex.jl setup

z = Variable(N) # decision variables
obj = (sum([quadform((z[idx_x[t]]-xF),sqrt(Q[t])) for t = 1:T]) # objective
        + sum([quadform(z[idx_u[t]],sqrt(R[t])) for t = 1:T-1]))

problem = minimize(obj) # setup problem

# constraint setup
for t = 1:T-1
    problem.constraints += z[idx_x[t+1]] - (A[t]*z[idx_x[t]] + B[t]*z[idx_u[t]]) == 0.0
end
problem.constraints += z[idx_x[1]] - x1 == 0.0
problem.constraints += z[idx_x[T]] - xF == 0.0

solve!(problem, ECOS.Optimizer) # solve

# Results
z_sol = z.value

# unpack solution
x_sol = [z_sol[idx_x[t][1]] for t = 1:T]
v_sol = [z_sol[idx_x[t][2]] for t = 1:T]
u_sol = [z_sol[idx_u[t]] for t = 1:T-1]

# state trajectory
plot(x_sol,v_sol,aspect_ratio=:equal,width=2.0,
    xlabel="position x", ylabel="velocity v",
    title="Double integrator state trajectory")
plot!([x_sol[1]],[v_sol[1]],marker=:circle,label="start",
    color=:red)
plot!([x_sol[T]],[v_sol[T]],marker=:circle,label="end",
    color=:green)

# control trajectory
plot(range(0,stop=Δt*T,length=T),hcat(u_sol...,u_sol[end])',width=2.0,linetype=:steppost,
    label="",xlabel="time (s)",title="Double integrator control (ZOH)")
