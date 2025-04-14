import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Define the function and its gradient
def f(x):
    return x**2 + 4*x + 4

def grad_f(x):
    return 2*x + 4

# Gradient Descent Method
def gradient_descent(x0, lr, method, max_iters=100):
    x = x0
    trajectory = [x]
    v = 0  # For momentum
    G = 0  # For Adagrad
    E = 0  # For RMSprop
    m = 0  # For Adam
    v_ = 0  # For Adam

    for t in range(1, max_iters + 1):
        grad = grad_f(x)
        
        if method == 'GD':
            x = x - lr * grad
        elif method == 'Momentum':
            v = 0.9 * v + (1 - 0.9) * grad
            x = x - lr * v
        elif method == 'Adagrad':
            G += grad**2
            x = x - lr / np.sqrt(G + 1e-8) * grad
        elif method == 'RMSprop':
            E = 0.9 * E + (1 - 0.9) * grad**2
            x = x - lr / np.sqrt(E + 1e-8) * grad
        elif method == 'Adam':
            m = 0.9 * m + (1 - 0.9) * grad
            v_ = 0.999 * v_ + (1 - 0.999) * grad**2
            m_hat = m / (1 - 0.9**t)
            v_hat = v_ / (1 - 0.999**t)
            x = x - lr * m_hat / (np.sqrt(v_hat) + 1e-8)

        trajectory.append(x)
        
    return np.array(trajectory)

# Initialize plot
fig, ax = plt.subplots(figsize=(10, 6))
x_vals = np.linspace(-5, 1, 400)
y_vals = f(x_vals)

ax.plot(x_vals, y_vals, label='Objective function: f(x)', color='black')
ax.axhline(0, color='black',linewidth=0.5)
ax.axvline(0, color='black',linewidth=0.5)
ax.set_xlim([-5, 1])
ax.set_ylim([0, 20])
ax.set_xlabel('x')
ax.set_ylabel('f(x)')
ax.set_title('Optimization Methods')

# Trajectory lines for each method
methods = ['GD', 'Momentum', 'Adagrad', 'RMSprop', 'Adam']
colors = ['r', 'b', 'g', 'm', 'c']  # Color for each method
lines = {method: ax.plot([], [], label=method, color=colors[i])[0] for i, method in enumerate(methods)}
markers = {method: ax.plot([], [], 'o', markersize=6, color=colors[i])[0] for i, method in enumerate(methods)}  # Matching marker color
trajectories = {method: [] for method in methods}

# Generate the trajectories for all methods
for method in methods:
    trajectories[method] = gradient_descent(5, 0.1, method)

# Update function for animation
def update(frame):
    for method in methods:
        line = lines[method]
        marker = markers[method]
        
        # Update the trajectory line
        line.set_data(trajectories[method][:frame], f(trajectories[method][:frame]))
        
        # Update the marker (latest position)
        marker.set_data(trajectories[method][frame-1], f(trajectories[method][frame-1]))
        
    return [line for line in lines.values()] + [marker for marker in markers.values()]

# Animate
ani = FuncAnimation(fig, update, frames=range(1, 100), interval=100, blit=True)

plt.legend()
plt.show()

