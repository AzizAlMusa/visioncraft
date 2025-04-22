import numpy as np
import matplotlib.pyplot as plt

# Define the piecewise function and its sigmoid approximation
def piecewise_function(coverage, qualities):
    """
    Pure piecewise step function
    """
    result = np.zeros_like(coverage)
    
    # Loop through the quality levels
    for i in range(len(qualities) - 1):
        level, value = qualities[i]
        next_level, next_value = qualities[i+1]
        
        # Set value for points within this range
        mask = (coverage >= level) & (coverage < next_level)
        result[mask] = value
    
    # Set value for points above the last threshold
    last_level, last_value = qualities[-1]
    result[coverage >= last_level] = last_value
    
    return result

def sigmoid_approximation(coverage, qualities, sharpness=12.0):
    """
    Sigmoid-based smooth approximation of the piecewise function
    """
    quality_values = np.zeros_like(coverage)
    
    # Apply each step
    for i in range(len(qualities)):
        level, value = qualities[i]
        
        if i == len(qualities) - 1:
            # Last level - apply to all points above this level
            weight = 1.0 / (1.0 + np.exp(-sharpness * (coverage - level)))
            quality_values += value * weight
        else:
            # Calculate window between this level and the next
            next_level = qualities[i+1][0]
            
            # Rising edge at current level
            rising_edge = 1.0 / (1.0 + np.exp(-sharpness * (coverage - level)))
            # Falling edge at next level
            falling_edge = 1.0 / (1.0 + np.exp(-sharpness * (coverage - next_level)))
            
            window = rising_edge * (1.0 - falling_edge)
            quality_values += value * window
    
    # Ensure quality values stay within [0, 1]
    return np.clip(quality_values, 0.0, 1.0)

# Sample quality configurations
# You can modify these as needed
qualities = [
    (0.0, 0.2),  # No coverage: value = 0.5
    (1.0, 0.0),  # Single coverage: value = 0.2
    (2.0, 1.0),  # Double coverage: value = 0.8
    (3.0, 1.0),  # Triple coverage: value = 1.0
    (4.0, 1.0)   # Four+ coverage: value = 1.0
]

# X-axis range for plotting
x = np.linspace(0, 5, 1000)

# Plot 1: Piecewise function
plt.figure(figsize=(8, 6))
y_piecewise = piecewise_function(x, qualities)
plt.plot(x, y_piecewise)
plt.xlabel('Coverage Level')
plt.ylabel('Visibility Value')
plt.title('Piecewise Visibility Function')
plt.grid(True, alpha=0.3)
plt.xlim(0, 5)  # Set x-axis to start at 0
plt.ylim(0, 1.1)  # Set y-axis to start at 0 with a bit of margin at top
# For the piecewise function plot
plt.savefig('piecewise_visibility.png', dpi=300, bbox_inches='tight')
plt.show()


# Plot 2: Sigmoid approximation
plt.figure(figsize=(8, 6))
y_sigmoid = sigmoid_approximation(x, qualities)
plt.plot(x, y_sigmoid)
plt.xlabel('Coverage Level')
plt.ylabel('Visibility Value')
plt.title('Sigmoid Approximation of Visibility Function')
plt.grid(True, alpha=0.3)
plt.xlim(0, 5)  # Set x-axis to start at 0
plt.ylim(0, 1.1)  # Set y-axis to start at 0 with a bit of margin at top
# For the sigmoid function plot
plt.savefig('sigmoid_visibility.png', dpi=300, bbox_inches='tight')
plt.show()
