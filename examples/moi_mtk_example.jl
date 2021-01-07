#NOTE: get latest version of ModelingToolkit.jl

using Plots
include(joinpath(pwd(),"src/moi_mtk.jl"))

# Problem
include(joinpath(pwd(),"src/problem.jl"))

# MOI setup

# objective function
function obj(z)
    # unpack decision variables
    x = [z[idx_x[t]] for t = 1:T]
    u = [z[idx_u[t]] for t = 1:T-1]

    J = 0.0
    for t = 1:T-1
        J += transpose(x[t] - xF)*Q[t]*(x[t] - xF)
        J += transpose(u[t])*R[t]*u[t]
    end
    J += transpose(x[T] - xF)*Q[T]*(x[T] - xF)

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

function lag(z,y)
    c0 = zeros(eltype(z),length(y))
    L = obj(z)

    x = [z[idx_x[t]] for t = 1:T]
    u = [z[idx_u[t]] for t = 1:T-1]

    for t = 1:T-1
        # dynamics
        for t = 1:T-1
            L += transpose(y[(t-1)*n .+ (1:n)])*(x[t+1] - (A[t]*x[t] + B[t]*u[t]))
        end

        # initial condition
        L += transpose(y[(T-1)*n .+ (1:n)])*(x[1] - x1)

        # final condition
        L += transpose(y[T*n .+ (1:n)])*(x[T] - xF)
    end

    return L
end

lag(rand(N),rand(M))

# generate fast functions
@variables x_sym[1:N], c_sym[1:M]
@parameters y_sym[1:M]

J = obj(x_sym);
obj_fast! = eval(ModelingToolkit.build_function([J],x_sym,
            parallel=ModelingToolkit.MultithreadedForm())[2])
∇obj_sparsity = ModelingToolkit.sparsejacobian([J],x_sym)
∇obj_fast! = eval(ModelingToolkit.build_function(∇obj_sparsity,x_sym,
            parallel=ModelingToolkit.MultithreadedForm())[2])
∇obj_fast = similar(∇obj_sparsity,Float64)

con!(c_sym,x_sym)
c_fast! = eval(ModelingToolkit.build_function(simplify.(c_sym),x_sym,
            parallel=ModelingToolkit.MultithreadedForm())[2])
∇c_sparsity = ModelingToolkit.sparsejacobian(simplify.(c_sym),x_sym)
∇c_fast! = eval(ModelingToolkit.build_function(∇c_sparsity,x_sym,
            parallel=ModelingToolkit.MultithreadedForm())[2])
∇c_fast = similar(∇c_sparsity,Float64)

dL = lag(x_sym,y_sym);

∇²L_sparsity = ModelingToolkit.sparsehessian(simplify.(dL),x_sym)
∇²L_fast! = eval(ModelingToolkit.build_function(∇²L_sparsity,x_sym,y_sym,
            parallel=ModelingToolkit.MultithreadedForm())[2])
∇²L_fast = similar(∇²L_sparsity,Float64)

moi_f = obj
moi_c! = con!

prob = ProblemMTK(N,M,
           (-Inf*ones(N),Inf*ones(N)),
           (zeros(M),zeros(M)),
           sparsity(∇c_sparsity),
           sparsity(∇²L_sparsity),
           ∇obj_fast,∇c_fast,∇²L_fast,
           true
           )

z0 = rand(N) # initialize w/ random decision variables

@time z_sol = solve(z0,prob) # solve

# Results

# unpack solution
x_sol = [z_sol[idx_x[t][1]] for t = 1:T]
v_sol = [z_sol[idx_x[t][2]] for t = 1:T]
u_sol = [z_sol[idx_u[t]] for t = 1:T-1]

# state trajectory
plt = plot(x_sol,v_sol,aspect_ratio=:equal,width=2.0,
    xlabel="position x", ylabel="velocity v",
    title="Double integrator state trajectory",
    label="trajectory")
plt = plot!([x_sol[1]],[v_sol[1]],marker=:circle,label="start",
    color=:red)
plt = plot!([x_sol[T]],[v_sol[T]],marker=:circle,label="end",
    color=:green)

savefig(plt,"/home/taylor/Research/thowell.github.io/images/simple_traj.png")

# control trajectory
plot(range(0,stop=Δt*T,length=T),
    hcat(u_sol...,u_sol[end])',width=2.0,linetype=:steppost,
    label="",xlabel="time (s)",title="Double integrator control (ZOH)")
