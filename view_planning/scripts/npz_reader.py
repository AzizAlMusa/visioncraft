import numpy as np

# Load the .npz file
data = np.load('./results2/sa_seed0_metrics.npz')

# Print the keys (names of arrays) stored inside
print("Keys in the npz file:", data.files)

# Optionally, print the content or shape of each array
for key in data.files:
    print(f"\nKey: {key}")
    print(f"Shape: {data[key].shape}")
    print(f"Data (first 10 elements): {data[key].flatten()[:10]}")


# Access the 'time' array
time_array = data['time']

# Print the last time value
final_time = time_array[-1]
print("Final time value:", final_time)

final_coverage = data['coverage'][-1]
print("Final coverage value:", final_coverage)