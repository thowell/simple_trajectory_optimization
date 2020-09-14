using Plots
include(joinpath(pwd(),"src/moi.jl"))

# Problem
include(joinpath(pwd(),"src/problem.jl"))

# MOI setup

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
    return c
end

prob = ProblemMOI(N,M)

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
    title="Double integrator state trajectory",
    label="trajectory")
plot!([x_sol[1]],[v_sol[1]],marker=:circle,label="start",
    color=:red)
plot!([x_sol[T]],[v_sol[T]],marker=:circle,label="end",
    color=:green)

# control trajectory
plot(range(0,stop=Δt*T,length=T),hcat(u_sol...,u_sol[end])',width=2.0,linetype=:steppost,
    label="",xlabel="time (s)",title="Double integrator control (ZOH)")
