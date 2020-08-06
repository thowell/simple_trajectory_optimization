using Ipopt, MathOptInterface
const MOI = MathOptInterface

struct ProblemIpopt <: MOI.AbstractNLPEvaluator
    n_nlp
    m_nlp
    obj
    ∇obj!
    con!
    ∇con!
    idx_ineq
    enable_hessian::Bool
    sparsity_jac
    reshape_jac::Bool
    primal_bounds
end

function ProblemIpopt(n_nlp,m_nlp,obj,con!,enable_hessian;
        idx_ineq=(1:0),
        sparsity_jac=sparsity_jacobian(n_nlp,m_nlp),
        reshape_jac=true,
        primal_bounds=primal_bounds(n_nlp))
    ∇obj!(g,z) = ForwardDiff.gradient!(g,obj,z)
    ∇con!(∇c,z) = ForwardDiff.jacobian!(∇c,con!,zeros(eltype(z),m_nlp),z)
    ProblemIpopt(n_nlp,m_nlp,obj,∇obj!,con!,∇con!,
        idx_ineq,
        enable_hessian,
        sparsity_jac,
        reshape_jac,
        primal_bounds)
end

function ProblemIpopt(n_nlp,m_nlp,obj,∇obj!,con!,∇con!,enable_hessian;
        idx_ineq=(1:0),
        sparsity_jac=sparsity_jacobian(n_nlp,m_nlp),
        reshape_jac=true,
        primal_bounds=primal_bounds(n_nlp))
    ProblemIpopt(n_nlp,m_nlp,obj,∇obj!,con!,∇con!,
        idx_ineq,
        enable_hessian,
        sparsity_jac,
        reshape_jac,
        primal_bounds)
end

function primal_bounds(n)
    x_l = -Inf*ones(n)
    x_u = Inf*ones(n)
    return x_l, x_u
end

function constraint_bounds(prob::MOI.AbstractNLPEvaluator)
    c_l = zeros(prob.m_nlp)
    c_l[prob.idx_ineq] .= -Inf
    c_u = zeros(prob.m_nlp)
    return c_l, c_u
end

function MOI.eval_objective(prob::MOI.AbstractNLPEvaluator, x)
    prob.obj(x)
end

function MOI.eval_objective_gradient(prob::MOI.AbstractNLPEvaluator, grad_f, x)
    prob.∇obj!(grad_f,x)
    return nothing
end

function MOI.eval_constraint(prob::MOI.AbstractNLPEvaluator,g,x)
    prob.con!(g,x)
    return nothing
end

function MOI.eval_constraint_jacobian(prob::MOI.AbstractNLPEvaluator, jac, x)
    # prob.∇con!(reshape(jac,prob.m_nlp,prob.n_nlp),x)
    prob.∇con!(jac,x)

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

function sparsity_jacobian(n,m)

    row = []
    col = []

    r = 1:m
    c = 1:n

    row_col!(row,col,r,c)

    return collect(zip(row,col))
end
sparsity_jacobian(prob::MOI.AbstractNLPEvaluator) = sparsity_jacobian(prob.n_nlp,prob.m_nlp)


function sparsity_hessian(prob::MOI.AbstractNLPEvaluator)

    row = []
    col = []

    r = 1:prob.n_nlp
    c = 1:prob.n_nlp

    row_col!(row,col,r,c)

    return collect(zip(row,col))
end

MOI.features_available(prob::MOI.AbstractNLPEvaluator) = [:Grad, :Jac]
MOI.initialize(prob::MOI.AbstractNLPEvaluator, features) = nothing
MOI.jacobian_structure(prob::MOI.AbstractNLPEvaluator) = prob.sparsity_jac
MOI.hessian_lagrangian_structure(prob::MOI.AbstractNLPEvaluator) = nothing
function MOI.eval_hessian_lagrangian(prob::MOI.AbstractNLPEvaluator, H, x, σ, λ)
    tmp(z) = σ*prob.obj(z) + prob.con!(zeros(eltype(z),prob.m_nlp),z)'*λ
    H .= vec(ForwardDiff.hessian(tmp,x))
    # println("eval hessian lagrangian")
    return nothing
end

function solve(x0,prob::MOI.AbstractNLPEvaluator;
        tol=1.0e-6,nlp=:ipopt,max_iter=1000)
    x_l, x_u = prob.primal_bounds
    c_l, c_u = constraint_bounds(prob)

    nlp_bounds = MOI.NLPBoundsPair.(c_l,c_u)
    block_data = MOI.NLPBlockData(nlp_bounds,prob,true)

    if nlp == :ipopt
        solver = Ipopt.Optimizer()
        solver.options["max_iter"] = max_iter
        solver.options["tol"] = tol
    elseif nlp == :snopt
        solver = SNOPT7.Optimizer()
    end

    x = MOI.add_variables(solver,prob.n_nlp)

    for i = 1:prob.n_nlp
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
    res = MOI.get(solver, MOI.VariablePrimal(), x)

    return res
end
