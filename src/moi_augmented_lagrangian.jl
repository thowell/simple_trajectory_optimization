using Ipopt, MathOptInterface
const MOI = MathOptInterface
using ForwardDiff

# user should define obj, con!
function obj(x)
    @error "objective not defined"
end

function con_eq!(c,x)
    @error "equality constraints not defined"
    return c
end

function con_ineq!(c,x)
    @error "inequality constraints not defined"
    return c
end

# user can overwrite primal_bounds, constraint_bounds
function primal_bounds(n)
    x_l = -Inf*ones(n)
    x_u = Inf*ones(n)
    return x_l, x_u
end

function ineq_constraint_bounds(m)
    c_l = -Inf*ones(m)
    c_u = zeros(m)
    return c_l, c_u
end

struct ALProblem <: MOI.AbstractNLPEvaluator
    n::Int
    m_ineq::Int
    m_eq::Int
    primal_bounds
    constraint_bounds
    λ
    ρ
    c_tol

    c_ineq
    c_eq
    ∇c_eq
end

function ALProblem(n,m_ineq,m_eq;
        primal_bounds=primal_bounds(n),
        constraint_bounds=ineq_constraint_bounds(m_ineq),
        ρ=[1.0],
        c_tol=1.0e-3)

    λ = zeros(m_eq)

    c_ineq = zeros(m_ineq)
    c_eq = zeros(m_eq)
    ∇c_eq = zeros(m_eq,n)

    ALProblem(n,m_ineq,m_eq,
              primal_bounds,
              constraint_bounds,
              λ,
              ρ,
              c_tol,
              c_ineq,
              c_eq,
              ∇c_eq)
end

function MOI.eval_objective(prob::ALProblem, x)
    c_eq = prob.c_eq
    con_eq!(c_eq,x)
    obj(x) + prob.λ'*c_eq + 0.5*prob.ρ[1]*c_eq'*c_eq
end

function MOI.eval_objective_gradient(prob::ALProblem, grad_f, x)
    c_eq = prob.c_eq
    ∇c_eq = prob.∇c_eq
    ForwardDiff.jacobian!(∇c_eq,con_eq!,c_eq,x)

    ForwardDiff.gradient!(grad_f,obj,x)
    grad_f .+= ∇c_eq'*(prob.λ + prob.ρ[1]*c_eq)
    return nothing
end

function MOI.eval_constraint(prob::ALProblem,g,x)
    con_ineq!(g,x)
    return nothing
end

function MOI.eval_constraint_jacobian(prob::ALProblem, jac, x)
    ForwardDiff.jacobian!(reshape(jac,prob.m_ineq,prob.n),
        con_ineq!,prob.c_ineq,x)
    return nothing
end

function row_col!(row,col,r,c)
    for cc in c
        for rr in r
            push!(row,convert(Int,rr))
            push!(col,convert(Int,cc))
        end
    end
    return row, col
end

function row_col_cartesian!(row,col,r,c)
    for i = 1:length(r)
        push!(row,convert(Int,r[i]))
        push!(col,convert(Int,c[i]))
    end
    return row, col
end

# user can overwrite sparsity_jacobian and sparsity_hessian
function sparsity_jacobian(n,m)

    row = []
    col = []

    r = 1:m
    c = 1:n

    row_col!(row,col,r,c)

    return collect(zip(row,col))
end

function sparsity_hessian(n,m)

    row = []
    col = []

    r = 1:m
    c = 1:n

    row_col!(row,col,r,c)

    return collect(zip(row,col))
end

function MOI.features_available(prob::ALProblem)
   return [:Grad, :Jac]
end
MOI.initialize(prob::ALProblem, features) = nothing
MOI.jacobian_structure(prob::ALProblem) = sparsity_jacobian(prob.n,prob.m_ineq)
MOI.hessian_lagrangian_structure(prob::ALProblem) = sparsity_hessian(prob.n,prob.m_ineq)
function MOI.eval_hessian_lagrangian(prob::ALProblem, H, x, σ, λ)
    tmp(z) = σ*obj(z) + con!(zeros(eltype(z),prob.m),z)'*λ

    H .= vec(ForwardDiff.hessian(tmp,x))
    return nothing
end

function gen_solver(x0,prob::ALProblem;
        tol=1.0e-6,c_tol=1.0e-6,nlp=:ipopt,max_iter=1000)
    x_l, x_u = prob.primal_bounds
    c_l, c_u = prob.constraint_bounds

    nlp_bounds = MOI.NLPBoundsPair.(c_l,c_u)
    block_data = MOI.NLPBlockData(nlp_bounds,prob,true)

    if nlp == :ipopt
        solver = Ipopt.Optimizer()
        solver.options["max_iter"] = max_iter
        solver.options["tol"] = tol
        solver.options["constr_viol_tol"] = c_tol
    elseif nlp == :snopt
        solver = SNOPT7.Optimizer()
        @warn "SNOPT user options not setup"
    else
        @error "NLP solver not setup"
    end

    x = MOI.add_variables(solver,prob.n)

    for i = 1:prob.n
        xi = MOI.SingleVariable(x[i])
        MOI.add_constraint(solver, xi, MOI.LessThan(x_u[i]))
        MOI.add_constraint(solver, xi, MOI.GreaterThan(x_l[i]))
        MOI.set(solver, MOI.VariablePrimalStart(), x[i], x0[i])
    end

    # Solve the problem
    MOI.set(solver, MOI.NLPBlock(), block_data)
    MOI.set(solver, MOI.ObjectiveSense(), MOI.MIN_SENSE)

    return solver, x
end

function solve(x0,prob::ALProblem;
        tol=1.0e-6,c_tol=1.0e-6,nlp=:ipopt,max_iter=1000)

    solver, x = gen_solver(x0,prob,c_tol=c_tol,tol=tol)

    max_iter = 10

    for i = 1:max_iter
        con_eq!(prob.c_eq,i==1 ? x0 : MOI.get(solver, MOI.VariablePrimal(), x))
        norm(prob.c_eq,Inf) < c_tol && break
        println("AL loop ($i/3)")

        if i < max_iter
            solver.options["tol"] = 1.0e-2
            solver.options["constr_viol_tol"] = 1.0e-2
        end

        if i > 1
            solver.options["warm_start_init_point"] = "yes"
            solver.options["mu_init"] = 1.0e-4
        end

        MOI.optimize!(solver)
        res = MOI.get(solver, MOI.VariablePrimal(), x)

        con_eq!(prob.c_eq,res)
        prob.λ .+= prob.ρ[1]*prob.c_eq
        prob.ρ[1] *= 10.0
    end

    return MOI.get(solver, MOI.VariablePrimal(), x)
end
