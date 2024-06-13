import numpy as np
from scipy.integrate import solve_ivp, solve_bvp
from scipy.optimize import root_scalar
from scipy.linalg import solve_banded
import matplotlib.pyplot as plt

# Define the ODE system for the Shooting and BVP methods
def odefunc(t, y, zeta=0.25):
    x, x_dot = y
    return [x_dot, -2*zeta*x_dot - x]

# Define the objective function for the Shooting method
def objective(shooting_guess, target):
    sol = solve_ivp(odefunc, [0, 9], [1, shooting_guess], t_eval=[9])
    return sol.y[0, -1] - target

# Shooting Method
result = root_scalar(objective, args=(0,), bracket=[-10, 10], method='bisect')
shooting_guess = result.root
t_eval = np.linspace(0, 9, 100)
sol_shooting = solve_ivp(odefunc, [0, 9], [1, shooting_guess], t_eval=t_eval)
x_shooting = sol_shooting.y[0]

# Finite Difference Method
N =100
t_fd = np.linspace(0, 9, N)
h = t_fd[1] - t_fd[0]
zeta = 0.25

# Construct the finite difference matrix
A = np.zeros((N, N))
B = np.zeros(N)
B[0] = 1  # Boundary condition at t=0
B[-1] = 0  # Boundary condition at t=9

# Fill the matrix A
for i in range(1, N-1):
    A[i, i-1] = 1/h**2 - zeta/(2*h)
    A[i, i] = -2/h**2 + 1
    A[i, i+1] = 1/h**2 + zeta/(2*h)

# Boundary conditions in the matrix
A[0, 0] = 1
A[-1, -1] = 1

# Solve the linear system
x_fd = np.linalg.solve(A, B)

# scipy.integrate.solve_bvp Method
def bc(ya, yb):
    return [ya[0] - 1, yb[0]]

y_init = np.zeros((2, t_eval.size))
y_init[0] = 1 - t_eval / 9

sol_bvp = solve_bvp(odefunc, bc, t_eval, y_init)
x_bvp = sol_bvp.sol(t_eval)[0]

# Plotting all solutions on one graph
plt.plot(t_eval, x_shooting, label='Shooting Method')
plt.plot(t_fd, x_fd, label='Finite Difference Method')
plt.plot(t_eval, x_bvp, label='scipy.integrate.solve_bvp Method')
plt.xlabel('t')
plt.ylabel('x(t)')
plt.title('Solutions to the Boundary Value Problem')
plt.legend()
plt.grid(True)
plt.show()
