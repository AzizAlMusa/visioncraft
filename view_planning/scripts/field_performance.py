import matplotlib.pyplot as plt
import pandas as pd

# Read the results.csv file into a pandas DataFrame
df = pd.read_csv('../build/results.csv')

# Extract the first 30 rows for the first 30 iterations
df = df.head(100)

# Extract the columns we need for the first 30 iterations
timesteps = df['Timestep']
coverage_score = df['CoverageScore']
system_energy = df['SystemEnergy']
kinetic_energy = df['KineticEnergy']
force_magnitude = df['ForceMagnitude']
entropy = df['Entropy']  # Assuming the entropy column is named 'VisibilityEntropy'

# Create a figure and axis for plotting with larger size and more space between subplots
fig, axs = plt.subplots(5, 1, figsize=(12, 24), sharex=True)

# Plot Coverage Score on the first subplot
axs[0].plot(timesteps, coverage_score, color='#1f77b4', linewidth=2)  # Use a good color (blue)
axs[0].set_ylabel('Coverage Score')
axs[0].set_title('Coverage Score Over Time')
axs[0].grid(True)

# Plot System Energy on the second subplot
axs[1].plot(timesteps, system_energy, color='#ff7f0e', linewidth=2)  # Use a good color (orange)
axs[1].set_ylabel('System Energy')
axs[1].set_title('System Energy Over Time')
axs[1].grid(True)

# Plot Kinetic Energy on the third subplot
axs[2].plot(timesteps, kinetic_energy, color='#2ca02c', linewidth=2)  # Use a good color (green)
axs[2].set_xlabel('Timestep')
axs[2].set_ylabel('Kinetic Energy')
axs[2].set_title('Kinetic Energy Over Time')
axs[2].grid(True)

# Plot Force Magnitude on the fourth subplot
axs[3].plot(timesteps, force_magnitude, color='#d62728', linewidth=2)  # Use a good color (red)
axs[3].set_xlabel('Timestep')
axs[3].set_ylabel('Force Magnitude')
axs[3].set_title('Force Magnitude Over Time')
axs[3].grid(True)

# Plot Entropy on the fifth subplot
axs[4].plot(timesteps, entropy, color='#9467bd', linewidth=2)  # Use a good color (purple)
axs[4].set_xlabel('Timestep')
axs[4].set_ylabel('Entropy')
axs[4].set_title('Visibility Entropy Over Time')
axs[4].grid(True)

# Adjust layout for better spacing
plt.tight_layout(pad=6.0)  # Increase padding between subplots for clarity

# Display the plot
plt.show()
