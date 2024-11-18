import numpy as np
import matplotlib.pyplot as plt

# Define parameters
D = 1  # Normalization constant (can be adjusted as per your model)
V_max = 100  # Example value for V_max

# Sample r values directly (distance r from 0 to 1000)
r_values = np.linspace(0, 1000, 500)  # r from 0 to 1000

# Compute the weight function W(r)
def W(r, D):
    return r**2 / D**2  # W(r) = r^2 / D^2

# Compute the gradient of the weight function
def grad_W(r, D):
    return 2 * r / D**2  # grad_W(r) = 2r / D^2

# Compute the attractive force (negative gradient)
def compute_attractive_force(r, V_norm, D):
    grad_W_value = grad_W(r, D)
    return -(1 - V_norm) * grad_W_value  # Attractive force

# Set V_norm (fixed for simplicity in this case)
V_norm = 0.5

# Compute W, grad_W, and F_attr
W_values = W(r_values, D)
grad_W_values = grad_W(r_values, D)
F_attr_values = compute_attractive_force(r_values, V_norm, D)

# Plot Weight function, Gradient, and Force for the given D value
plt.figure(figsize=(10, 12))

# Plot Weight Function W(r)
plt.subplot(3, 1, 1)
plt.plot(r_values, W_values, label='$W(r) = r^2 / D^2$')
plt.title('Weight Function $W(r)$')
plt.xlabel('Distance (r)')
plt.ylabel('W(r)')
plt.grid(True)
plt.legend()

# Plot Gradient of W(r)
plt.subplot(3, 1, 2)
plt.plot(r_values, grad_W_values, label='$\nabla W(r) = 2r / D^2$', color='r')
plt.title('Gradient of Weight Function $\nabla W(r)$')
plt.xlabel('Distance (r)')
plt.ylabel('grad_W(r)')
plt.grid(True)
plt.legend()

# Plot Attractive Force F_attr(r)
plt.subplot(3, 1, 3)
plt.plot(r_values, F_attr_values, label='$F_{{attr}}(r) = -(1 - V_{{norm}}) \nabla W(r)$', color='g')
plt.title('Attractive Force $F_{{attr}}(r)$')
plt.xlabel('Distance (r)')
plt.ylabel('Force Magnitude')
plt.grid(True)
plt.legend()

# Adjust layout and show plot
plt.tight_layout()
plt.show()
