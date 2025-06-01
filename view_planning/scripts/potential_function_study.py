import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.colors as mcolors

# Set up the figure with better spacing and modern design
plt.rcParams.update({
    'font.size': 10,
    'font.family': 'sans-serif',
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

# Create a larger figure with 3 rows and 5 columns
fig = plt.figure(figsize=(20, 15))
gs = GridSpec(3, 5, height_ratios=[0.8, 1.1, 1.1], hspace=0.35, wspace=0.3)

# Define potential functions and their gradients (forces)
potential_types = ["logarithmic", "linear", "quadratic", "inverse", "gaussian"]

# Distance range for visualization
r = np.linspace(0.1, 10, 1000)  # Avoid r=0 for potential functions that have singularities
r_2d = np.linspace(-10, 10, 100)
X, Y = np.meshgrid(r_2d, r_2d)
R = np.sqrt(X**2 + Y**2) + 1e-10  # Added small value to avoid division by zero

# Parameters
sigma = 5.0  # For Gaussian potential
alpha = 1.0  # Scaling factor

# Function to compute potentials
def compute_potential(r, potential_type):
    if potential_type == "logarithmic":
        return alpha * np.log(r)
    elif potential_type == "linear":
        return alpha * r
    elif potential_type == "quadratic":
        return alpha * r**2
    elif potential_type == "inverse":
        return -alpha / r  # Added negative sign
    elif potential_type == "gaussian":
        return -alpha * np.exp(-r**2 / sigma**2)  # Added negative sign

# Function to compute force magnitudes (positive gradient of potential for gradient ascent)
def compute_force(r, potential_type):
    if potential_type == "logarithmic":
        return alpha / r  # Positive gradient
    elif potential_type == "linear":
        return alpha * np.ones_like(r)  # Constant force
    elif potential_type == "quadratic":
        return 2 * alpha * r
    elif potential_type == "inverse":
        return alpha / r**2  # Positive gradient of negative potential
    elif potential_type == "gaussian":
        return 2 * alpha * r * np.exp(-r**2 / sigma**2) / sigma**2  # Positive gradient of negative potential

# Function to compute 2D potential field
def compute_potential_2d(X, Y, potential_type):
    R = np.sqrt(X**2 + Y**2) + 1e-10  # Avoid division by zero
    return compute_potential(R, potential_type)

# Function to compute 2D force field with scaled visualization
def compute_force_2d(X, Y, potential_type):
    R = np.sqrt(X**2 + Y**2) + 1e-10
    force_magnitude = compute_force(R, potential_type)
    
    # Unit vectors in radial direction
    X_norm = X / R
    Y_norm = Y / R
    
    return force_magnitude * X_norm, force_magnitude * Y_norm

# Define custom scaling factors for quiver plots to ensure consistent visibility
quiver_scales = {
    "logarithmic": 15,
    "linear": 30,
    "quadratic": 80, 
    "inverse": 8,
    "gaussian": 5
}

# Define quiver colors based on background brightness
quiver_colors = {
    "logarithmic": "white",
    "linear": "#1A365D",  # Darker blue for light background
    "quadratic": "white",
    "inverse": "white",
    "gaussian": "#0000ff"  # Bright blue for better contrast
}

# Consistent modern colors for all plots
potential_color = "#6200EA"  # More purplish blue
force_color = "#FF4081"      # More pinkish red

# Line styles with higher contrast
line_styles = {
    "potential": {"linestyle": '-', "linewidth": 2.5},
    "force": {"linestyle": '--', "linewidth": 2.5, "dashes": (5, 2)}
}

# Mathematical formulas for display (updated with negative signs)
formulas = {
    "logarithmic": r"$\Phi(r) = \alpha \log(r)$", 
    "linear": r"$\Phi(r) = \alpha r$",
    "quadratic": r"$\Phi(r) = \alpha r^2$",
    "inverse": r"$\Phi(r) = -\frac{\alpha}{r}$",  # Added negative sign
    "gaussian": r"$\Phi(r) = -\alpha e^{-r^2/\sigma^2}$"  # Added negative sign
}

force_formulas = {
    "logarithmic": r"$\vec{F} = \frac{\alpha}{r}\hat{r}$",
    "linear": r"$\vec{F} = \alpha\hat{r}$",
    "quadratic": r"$\vec{F} = 2\alpha r\hat{r}$",
    "inverse": r"$\vec{F} = \frac{\alpha}{r^2}\hat{r}$",  # Positive gradient of negative potential
    "gaussian": r"$\vec{F} = \frac{2\alpha r}{\sigma^2}e^{-r^2/\sigma^2}\hat{r}$"  # Positive gradient of negative potential
}

# ===== 1D POTENTIAL AND FORCE PLOTS (ONE FOR EACH TYPE) =====
ax_1d = {}
for i, pot_type in enumerate(potential_types):
    ax_1d[pot_type] = fig.add_subplot(gs[0, i])
    ax_1d[pot_type].set_title(f"{pot_type.capitalize()} Function", fontsize=14)
    ax_1d[pot_type].set_xlabel("Distance (r)", fontsize=12)
    
    if i == 0:  # Only add y-label to the first plot
        ax_1d[pot_type].set_ylabel("Normalized Value", fontsize=12)
    
    ax_1d[pot_type].grid(True, alpha=0.3)
    
    # Calculate potential values
    potential_values = compute_potential(r, pot_type)
    
    # Normalize for better visualization
    if pot_type != "inverse" and pot_type != "logarithmic":
        potential_values = potential_values / np.max(np.abs(potential_values))
    elif pot_type == "inverse":
        # Cap inverse potential for visualization (now negative)
        potential_values = np.maximum(potential_values, -3)
    elif pot_type == "logarithmic":
        # Normalize logarithmic potential
        potential_values = (potential_values - np.min(potential_values)) / (np.max(potential_values) - np.min(potential_values))
    
    # Plot potential
    ax_1d[pot_type].plot(r, potential_values, 
                        label="Potential", 
                        color=potential_color, 
                        **line_styles["potential"])
    
    # Calculate force values
    force_values = compute_force(r, pot_type)
    
    # Normalize for better visualization
    if pot_type != "inverse" and pot_type != "gaussian":
        force_values = force_values / np.max(np.abs(force_values))
    elif pot_type == "inverse":
        # Cap inverse force for visualization
        force_values = np.maximum(force_values, -3)
    elif pot_type == "gaussian":
        # Normalize Gaussian force
        force_values = force_values / np.max(np.abs(force_values))
    
    # Plot force
    ax_1d[pot_type].plot(r, force_values, 
                        label="Force", 
                        color=force_color, 
                        **line_styles["force"])
    
    # Add legend to each plot
    ax_1d[pot_type].legend(loc='best', fontsize=10)
    
    # Set consistent y-limits across all plots
    ax_1d[pot_type].set_ylim(-3.2, 3.2)

# ===== 2D POTENTIAL FIELDS =====
ax_potentials = {}
for i, pot_type in enumerate(potential_types):
    ax_potentials[pot_type] = fig.add_subplot(gs[1, i])
    
    # Add formula to the title
    ax_potentials[pot_type].set_title(f"{pot_type.capitalize()} Potential\n{formulas[pot_type]}", fontsize=14)
    ax_potentials[pot_type].set_xlabel("x", fontsize=12)
    
    if i == 0:  # Only add y-label to the first plot
        ax_potentials[pot_type].set_ylabel("y", fontsize=12)
    
    # Compute 2D potential field
    potential_field = compute_potential_2d(X, Y, pot_type)
    
    # Handle special cases for visualization
    if pot_type == "inverse":
        # Cap the inverse potential (now negative, so use maximum)
        potential_field = np.maximum(potential_field, -3)
    elif pot_type == "logarithmic":
        # Handle negative values in logarithmic potential
        potential_field = np.maximum(potential_field, -3)
    
    # Plot filled contour
    contour = ax_potentials[pot_type].contourf(
        X, Y, potential_field, 
        levels=20, cmap='viridis', alpha=0.9
    )
    
    # Add colorbar with specific size and position
    cbar = plt.colorbar(contour, ax=ax_potentials[pot_type], fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=9)
    
    # Fix aspect ratio to be equal
    ax_potentials[pot_type].set_aspect('equal')

# ===== 2D FORCE FIELDS =====
ax_forces = {}
for i, pot_type in enumerate(potential_types):
    ax_forces[pot_type] = fig.add_subplot(gs[2, i])
    
    # Add formula to the title
    ax_forces[pot_type].set_title(f"{pot_type.capitalize()} Force Field\n{force_formulas[pot_type]}", fontsize=14)
    ax_forces[pot_type].set_xlabel("x", fontsize=12)
    
    if i == 0:  # Only add y-label to the first plot
        ax_forces[pot_type].set_ylabel("y", fontsize=12)
    
    # Compute 2D force field
    fx, fy = compute_force_2d(X, Y, pot_type)
    
    # Compute force magnitude
    force_magnitude = np.sqrt(fx**2 + fy**2)
    
    # Handle special cases for visualization
    if pot_type == "inverse":
        # Cap the inverse force magnitude
        force_magnitude = np.minimum(force_magnitude, 3)
        
    # Normalize force magnitude for colormap
    if np.max(force_magnitude) > 0:
        force_magnitude = force_magnitude / np.max(force_magnitude)
    
    # Plot force magnitude as colormap
    img = ax_forces[pot_type].imshow(
        force_magnitude.T, origin='lower', 
        extent=[-10, 10, -10, 10],
        cmap='inferno', alpha=0.85
    )
    
    # Add colorbar with better sizing
    cbar = plt.colorbar(img, ax=ax_forces[pot_type], fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=9)
    cbar.set_label("Force Magnitude", fontsize=10)
    
    # Set epsilon for excluding quivers too close to the center
    epsilon = 1.2  # Exclusion radius
    
    # Plot force vectors with custom scaling factors and consistent downsampling
    step = 7  # Consistent downsampling for all plots
    
    # Create mask for points outside epsilon (but don't visualize the epsilon)
    mask = R[::step, ::step] > epsilon
    
    # Apply mask to get coordinates and forces only for points outside epsilon
    x_points = X[::step, ::step][mask]
    y_points = Y[::step, ::step][mask]
    fx_points = fx[::step, ::step][mask]
    fy_points = fy[::step, ::step][mask]
    
    # Draw quivers using masked points with increased zorder for gaussian
    z_order = 10 if pot_type == "gaussian" else 5
    ax_forces[pot_type].quiver(
        x_points, y_points,
        fx_points, fy_points,
        color=quiver_colors[pot_type], scale=quiver_scales[pot_type], 
        width=0.004, pivot='mid',
        headwidth=4, headlength=5, zorder=z_order
    )
    
    # Add central particle
    ax_forces[pot_type].scatter([0], [0], c='cyan', s=120, edgecolor='white', zorder=6)
    
    # Fix aspect ratio to be equal
    ax_forces[pot_type].set_aspect('equal')

# Adjust layout for the entire figure
plt.subplots_adjust(top=0.96, bottom=0.05, left=0.05, right=0.95, hspace=0.4, wspace=0.3)

plt.show()

# Uncomment to save
# plt.savefig('corrected_potential_field_analysis.png', dpi=300, bbox_inches='tight')