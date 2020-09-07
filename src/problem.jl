using LinearAlgebra

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
