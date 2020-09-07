using LinearAlgebra, ModelingToolkit, ForwardDiff, SparseArrays, StaticArrays
using Ipopt
using MathOptInterface
const MOI = MathOptInterface

struct ProblemMTK <: MOI.AbstractNLPEvaluator
    n
    m
    primal_bounds
    constraint_bounds
    jacobian_sparsity
    hessian_sparsity
    ∇obj_fast
    ∇c_fast
    ∇²L_fast
    enable_hessian::Bool
end

function ProblemMTK(n,m,
        primal_bounds,
        constraint_bounds,
        jacobian_sparsity,
        hessian_sparsity,
        ∇obj_fast,∇c_fast,∇²L_fast;
        enable_hessian=true)

    ProblemMTK(n,m,
            primal_bounds,
            constraint_bounds,
            jacobian_sparsity,
            hessian_sparsity,
            ∇obj_fast,∇c_fast,∇²L_fast,
            enable_hessian)
end

function MOI.eval_objective(prob::MOI.AbstractNLPEvaluator, x)
    moi_f(x)
end

function MOI.eval_objective_gradient(prob::MOI.AbstractNLPEvaluator, grad_f, x)
    ∇obj_fast!(prob.∇obj_fast,x)
    grad_f .= vec(prob.∇obj_fast)
    return nothing
end

function MOI.eval_constraint(prob::MOI.AbstractNLPEvaluator,g,x)
    moi_c!(g,x)
    return nothing
end

function MOI.eval_constraint_jacobian(prob::MOI.AbstractNLPEvaluator, jac, x)
    ∇c_fast!(prob.∇c_fast,x)
    jac .= prob.∇c_fast.nzval
    return nothing
end

function MOI.eval_hessian_lagrangian(prob::MOI.AbstractNLPEvaluator, H, x, σ, λ)
    ∇²L_fast!(prob.∇²L_fast,x,λ)
    H .= prob.∇²L_fast.nzval

    return nothing
end

function primal_bounds(n)
    x_l = -Inf*ones(n)
    x_u = Inf*ones(n)
    return x_l, x_u
end

function constraint_bounds(m; idx_ineq=(1:0))
    c_l = zeros(m)
    c_l[idx_ineq] .= -Inf
    c_u = zeros(m)
    return c_l, c_u
end

function sparsity(x)
    (row,col,val) = findnz(x)
    collect(zip(row,col))
end

function MOI.features_available(prob::MOI.AbstractNLPEvaluator)
    if prob.enable_hessian
        return [:Grad, :Jac, :Hess]
    else
        return [:Grad, :Jac]
    end
end
MOI.initialize(prob::MOI.AbstractNLPEvaluator, features) = nothing
MOI.jacobian_structure(prob::MOI.AbstractNLPEvaluator) = prob.jacobian_sparsity
MOI.hessian_lagrangian_structure(prob::MOI.AbstractNLPEvaluator) = prob.hessian_sparsity

function solve(x0,prob::MOI.AbstractNLPEvaluator;
        tol=1.0e-6,c_tol=1.0e-6,max_iter=1000,nlp=:ipopt)

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

    MOI.optimize!(solver)

    # Get the solution
    sol = MOI.get(solver, MOI.VariablePrimal(), x)

    return sol
end
