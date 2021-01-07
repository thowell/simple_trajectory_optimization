using Convex, ECOS, Plots

# Problem
include(joinpath(pwd(),"src/problem.jl"))

# Convex.jl setup

z = Variable(N) # decision variables
objective = (sum([quadform((z[idx_x[t]]-xF),sqrt(Q[t])) for t = 1:T]) # objective
        + sum([quadform(z[idx_u[t]],sqrt(R[t])) for t = 1:T-1]))

problem = minimize(objective) # setup problem

# constraint setup
for t = 1:T-1
    (problem.constraints
        += z[idx_x[t+1]] - (A[t]*z[idx_x[t]] + B[t]*z[idx_u[t]]) == 0.0)
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
    title="Double integrator state trajectory",
    label="trajectory")
plot!([x_sol[1]],[v_sol[1]],marker=:circle,label="start",
    color=:red)
plot!([x_sol[T]],[v_sol[T]],marker=:circle,label="end",
    color=:green)

# control trajectory
plot(range(0,stop=Δt*T,length=T),
    hcat(u_sol...,u_sol[end])',width=2.0,linetype=:steppost,
    label="",xlabel="time (s)",title="Double integrator control (ZOH)")
