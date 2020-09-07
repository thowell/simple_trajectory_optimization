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

# objective gradient function
function ∇obj!(g,z)
    # unpack decision variables
    x = [z[idx_x[t]] for t = 1:T]
    u = [z[idx_u[t]] for t = 1:T-1]

    for t = 1:T
        g[idx_x[t]] = 2.0*Q[t]*(x[t] - xF)
        t==T && continue # cost on u only from t=1:T-1
        g[idx_u[t]] = 2.0*R[t]*u[t]
    end
    return g
end

# constraint Jacobian function
function ∇con!(j,z)

    shift = 0
    jac_shift = 0

    # dynamics
    for t = 1:T-1
        # c[shift .+ (1:n)] = x[t+1] - (A[t]*x[t] + B[t]*u[t])
        r_idx = shift .+ (1:n)

        c_idx = idx_x[t]
        len = length(r_idx)*length(c_idx)
        j[jac_shift .+ (1:len)] = vec(-1.0*A[t])
        jac_shift += len

        c_idx = idx_u[t]
        len = length(r_idx)*length(c_idx)
        j[jac_shift .+ (1:len)] = vec(-1.0*B[t])
        jac_shift += len

        c_idx = idx_x[t+1]
        len = n
        j[jac_shift .+ (1:len)] .= 1.0
        jac_shift += len

        shift += n
    end

    # initial condition
    # c[shift .+ (1:n)] = x[1] - x1
    r_idx = shift .+ (1:n)

    c_idx = idx_x[1]
    len = n
    j[jac_shift .+ (1:len)] .= 1.0
    jac_shift += len

    shift += n

    # final condition
    # c[shift .+ (1:n)] = x[T] - xF

    r_idx = shift .+ (1:n)

    c_idx = idx_x[T]
    len = n
    j[jac_shift .+ (1:len)] .= 1.0
    jac_shift += len

    shift += n

    return j
end

# sparsity structure of constraint Jacobian
function sparsity_jacobian(n,m,T,idx_x,idx_u)

    row = []
    col = []

    shift = 0
    jac_shift = 0

    # dynamics
    for t = 1:T-1
        # c[shift .+ (1:n)] = x[t+1] - (A[t]*x[t] + B[t]*u[t])
        r_idx = shift .+ (1:n)

        c_idx = idx_x[t]
        row_col!(row,col,r_idx,c_idx)

        c_idx = idx_u[t]
        row_col!(row,col,r_idx,c_idx)

        c_idx = idx_x[t+1]
        row_col_cartesian!(row,col,r_idx,c_idx)

        shift += n
    end

    # initial condition
    # c[shift .+ (1:n)] = x[1] - x1
    r_idx = shift .+ (1:n)
    c_idx = idx_x[1]
    row_col_cartesian!(row,col,r_idx,c_idx)

    shift += n

    # final condition
    # c[shift .+ (1:n)] = x[T] - xF

    r_idx = shift .+ (1:n)
    c_idx = idx_x[T]
    row_col_cartesian!(row,col,r_idx,c_idx)
    shift += n

    return collect(zip(row,col))
end

# Verify sparse methods
z0_test = rand(N)
g0 = zeros(N)
@assert norm(∇obj!(g0,z0_test) - ForwardDiff.gradient(obj,z0_test)) < 1.0e-8

c_sparsity = sparsity_jacobian(n,m,T,idx_x,idx_u)
j0 = zeros(M,N)
j0_vec = zeros(length(c_sparsity))
∇con!(j0_vec,z0_test)
for (i,idx) in enumerate(c_sparsity)
    j0[idx[1],idx[2]] = j0_vec[i]
end
@assert norm(vec(j0) - vec(ForwardDiff.jacobian(con!,zeros(M),z0_test))) < 1.0e-8

prob = ProblemMOI(N,M)
prob_sparse = ProblemMOI(N,M,
    obj_grad=true,
    con_jac=true,
    sparsity_jac=sparsity_jacobian(n,m,T,idx_x,idx_u))

z0 = rand(N) # initialize w/ random decision variables

@time z_sol = solve(z0,prob) # solve
@time z_sol = solve(z0,prob_sparse) # solve sparse problem

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
