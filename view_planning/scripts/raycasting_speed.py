import matplotlib.pyplot as plt

# Data
resolutions = ['50x50', '100x100', '200x200', '300x300', '400x400', '500x500', '1000x1000', '1500x1500', '2000x2000', '2500x2500']
cpu_times = [163.41, 686.75, 2641.24, 6012.91, 10638.20, 16341.50, 64904.70, 145800.00, 261410.00, 405842.00]
cpu_mt_times = [35.07, 175.68, 553.30, 1163.25, 2058.87, 3204.81, 11993.50, 28024.80, 49845.80, 77359.80]
gpu_times = [2.98, 6.12, 11.76, 21.29, 30.80, 40.23, 72.47, 110.72, 170.29, 247.55]

# Plot
plt.figure(figsize=(10, 6))

plt.plot(resolutions, cpu_times, label='CPU', marker='o', color='b', linestyle='-', linewidth=2)
plt.plot(resolutions, cpu_mt_times, label='CPU + Multithreading', marker='s', color='g', linestyle='--', linewidth=2)
plt.plot(resolutions, gpu_times, label='GPU', marker='^', color='r', linestyle='-.', linewidth=2)

# Labels and Title
plt.xlabel('Resolution', fontsize=12)
plt.ylabel('Time (ms)', fontsize=12)
plt.title('Comparison of Methods for Raycasting', fontsize=14)

# Set log scale for better visualization of the time differences
plt.yscale('log')

# Add grid, legend, and tight layout
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.legend()
plt.tight_layout()

# Show plot
plt.show()
