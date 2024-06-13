import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from scipy.integrate import solve_ivp

# Define the damped harmonic oscillator differential equations
def f_explicit(x, y, zeta):
    return -2 * zeta * y - x

def f_implicit(y_next, x_curr, y_curr, zeta, dt):
    return y_next - y_curr - dt * (-2 * zeta * y_next - x_curr)

def damped_oscillator(t, z, zeta):
    x, y = z
    return [y, -2 * zeta * y - x]

# Explicit Euler Method
def explicit_euler(f, x0, y0, t0, t1, dt, zeta):
    t = np.arange(t0, t1, dt)
    x = np.zeros(t.shape)
    y = np.zeros(t.shape)
    x[0], y[0] = x0, y0

    for i in range(1, len(t)):
        x[i] = x[i-1] + dt * y[i-1]
        y[i] = y[i-1] + dt * f(x[i-1], y[i-1], zeta)
    
    return t, x, y

# Implicit Euler Method
def implicit_euler(f, x0, y0, t0, t1, dt, zeta):
    t = np.arange(t0, t1, dt)
    x = np.zeros(t.shape)
    y = np.zeros(t.shape)
    x[0], y[0] = x0, y0

    for i in range(1, len(t)):
        def equations(vars):
            yi = vars[0]
            return f_implicit(yi, x[i-1], y[i-1], zeta, dt)
        
        yi = fsolve(equations, y[i-1])[0]
        xi = x[i-1] + dt * yi
        x[i], y[i] = xi, yi
    
    return t, x, y

# Calculate errors compared to solve_ivp
def calculate_errors(sol_ivp, t, x, y):
    sol_x = sol_ivp.sol(t)[0]
    sol_y = sol_ivp.sol(t)[1]
    error_x = np.abs(x - sol_x)
    error_y = np.abs(y - sol_y)
    return t, error_x, error_y
# Main function to generate plots for each zeta
def generate_plots(zeta_values, t0, t1, dt_values):
    for zeta in zeta_values:
        global fig, axs
        fig, axs =  plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Damped Harmonic Oscillator for zeta={zeta}', fontsize=16)
        
        for dt in dt_values:
            # Explicit Euler
            t_explicit, x_explicit, y_explicit = explicit_euler(f_explicit, 1, 0, t0, t1, dt, zeta)
            
            # Implicit Euler
            t_implicit, x_implicit, y_implicit = implicit_euler(f_implicit, 1, 0, t0, t1, dt, zeta)
            
            # SciPy solve_ivp
            sol_ivp = solve_ivp(damped_oscillator, [t0, t1], [1, 0], args=(zeta,), dense_output=True)
            t_ivp = np.linspace(t0, t1, len(t_explicit))
            sol = sol_ivp.sol(t_ivp)
            
            # Velocity over coordinate
            axs[0, 0].plot(x_explicit, y_explicit, label=f'Explicit Euler (dt={dt})')
            axs[0, 0].plot(x_implicit, y_implicit, label=f'Implicit Euler (dt={dt})')
            axs[0, 0].plot(sol[0], sol[1], '--', label=f'SciPy solve_ivp (dt={dt})')
            
            # Velocity over time
            axs[0, 1].plot(t_explicit, y_explicit, label=f'Explicit Euler (dt={dt})')
            axs[0, 1].plot(t_implicit, y_implicit, label=f'Implicit Euler (dt={dt})')
            axs[0, 1].plot(t_ivp, sol[1], '--', label=f'SciPy solve_ivp (dt={dt})')
            
            # Coordinate over time
            axs[1, 0].plot(t_explicit, x_explicit, label=f'Explicit Euler (dt={dt})')
            axs[1, 0].plot(t_implicit, x_implicit, label=f'Implicit Euler (dt={dt})')
            axs[1, 0].plot(t_ivp, sol[0], '--', label=f'SciPy solve_ivp (dt={dt})')
        max_errors_implicit=[]
        max_errors_explicit=[]
        for dt in dt_values1:
            # Explicit Euler
            t_explicit, x_explicit, y_explicit = explicit_euler(f_explicit, 1, 0, t0, t1, dt, zeta)
            
            # Implicit Euler
            t_implicit, x_implicit, y_implicit = implicit_euler(f_implicit, 1, 0, t0, t1, dt, zeta)
            # SciPy solve_ivp
            sol_ivp = solve_ivp(damped_oscillator, [t0, t1], [1, 0], args=(zeta,), dense_output=True)
            t_ivp = np.linspace(t0, t1, len(t_explicit))
            sol = sol_ivp.sol(t_ivp)
            # Error over time steps
            t_error_explicit, error_x_explicit, error_y_explicit = calculate_errors(sol_ivp, t_explicit, x_explicit, y_explicit)
            t_error_implicit, error_x_implicit, error_y_implicit = calculate_errors(sol_ivp, t_implicit, x_implicit, y_implicit)
            max_errors_implicit=np.append(max_errors_implicit, np.max(error_x_implicit))
            max_errors_explicit=np.append(max_errors_explicit, np.max(error_x_explicit))
        
        axs[1, 1].loglog(dt_values1, max_errors_explicit, label=f'Error Explicit Euler x (t={10})') 
        axs[1, 1].loglog(dt_values1, max_errors_implicit, label=f'Error Implicit Euler x (t={10})')   
               
            
        
        # Formatting plots
        axs[0, 0].set_title('Velocity over Coordinate')
        axs[0, 0].set_xlabel('x')
        axs[0, 0].set_ylabel('y')
        axs[0, 0].legend()
        axs[0, 0].grid(True)
        
        axs[0, 1].set_title('Velocity over Time')
        axs[0, 1].set_xlabel('Time')
        axs[0, 1].set_ylabel('Velocity (y)')
        axs[0, 1].legend()
        axs[0, 1].grid(True)
        
        axs[1, 0].set_title('Coordinate over Time')
        axs[1, 0].set_xlabel('Time')
        axs[1, 0].set_ylabel('Coordinate (x)')
        axs[1, 0].legend()  
        axs[1, 0].grid(True)
        
        axs[1, 1].set_title('Error over Time Steps')
        axs[1, 1].set_xlabel('Time')
        axs[1, 1].set_ylabel('Error')
        axs[1, 1].legend()
        axs[1, 1].grid(True)
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

# Define parameters
zeta_values = [0.25,1,2]
t0, t1 = 0, 10
dt_values = [0.05, 0.3]
dt_values1=np.linspace(0.001, 9.99, 1000)

    

# Generate plots
generate_plots(zeta_values, t0, t1, dt_values)
