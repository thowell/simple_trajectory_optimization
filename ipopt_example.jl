using LinearAlgebra, Plots
include("src/ipopt.jl")

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

# Ipopt setup

# objective function
function obj(z)
    # unpack decision variables
    x = [z[idx_x[t]] for t = 1:T]
    u = [z[idx_u[t]] for t = 1:T-1]

    J = 0.0
    for t = 1:T
        J += (x[t] - xF)'*Q[t]*(x[t] - xF)
        t==T && continue # cost on u only from t=1:T-1
        J += u[t]'*R[t]*u[t]
    end
    return J
end

# constraint function
function con!(c,z)
    # unpack decision variables
    x = [z[idx_x[t]] for t = 1:T]
    u = [z[idx_u[t]] for t = 1:T-1]

    shift = 0 # shift index used for convenience

    # dynamics
    for t = 1:T-1
        c[shift .+ (1:n)] = x[t+1] - (A[t]*x[t] + B[t]*u[t])
        shift += n
    end

    # initial condition
    c[shift .+ (1:n)] = x[1] - x1
    shift += n

    # final condition
    c[shift .+ (1:n)] = x[T] - xF
    return nothing
end

N = n*T + m*(T-1) # number of decision variables
M = n*(T+1) # number of constraints

prob = ProblemIpopt(N,M) # set up optimization problem for Ipopt

z0 = rand(N) # initialize w/ random decision variables

z_sol = solve(z0,prob) # solve

# Results

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
