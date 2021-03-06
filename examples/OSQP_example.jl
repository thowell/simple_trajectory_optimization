using OSQP, Plots, SparseArrays

# Problem
include(joinpath(pwd(),"src/problem.jl"))

# OSQP setup
P = spzeros(N,N)
q = zeros(N)
_A = spzeros(M,N)
l = zeros(M)
u = zeros(M)

# P
for t = 1:T
    P[idx_x[t],idx_x[t]] = Q[t]
    t==T && continue
    P[idx_u[t],idx_u[t]] = R[t]
end

# q
for t = 1:T
    q[idx_x[t]] = -2.0*Q[t]*xF
end

# A
for t = 1:T-1
    _A[(t-1)*n .+ (1:n),idx_x[t]] = -A[t]
    _A[(t-1)*n .+ (1:n),idx_u[t]] = -B[t]
    _A[(t-1)*n .+ (1:n),idx_x[t+1]] = Diagonal(ones(n))
end
_A[(T-1)*n .+ (1:n),idx_x[1]] = Diagonal(ones(n))
_A[(T)*n .+ (1:n),idx_x[T]] = Diagonal(ones(n))

# l, u
l[(T-1)*n .+ (1:n)] = x1
u[(T-1)*n .+ (1:n)] = x1
l[(T)*n .+ (1:n)] = xF
u[(T)*n .+ (1:n)] = xF

model = OSQP.Model()
OSQP.setup!(model,P=P,q=q,A=_A,l=l,u=u,linsys_solver="mkl_pardiso",polish=true)
res = OSQP.solve!(model)

# Results
z_sol = res.x

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
