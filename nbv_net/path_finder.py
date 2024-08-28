import os

# Define SCRIPT_DIR as the directory containing this script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Construct paths
scale_factor_file = os.path.normpath(os.path.join(SCRIPT_DIR, '..', 'view_planning', 'data', 'scaling_factor.txt'))
object_model_path = os.path.normpath(os.path.join(SCRIPT_DIR, '..', 'view_planning', 'model', 'gorilla.ply'))
cpp_simulator_path = os.path.normpath(os.path.join(SCRIPT_DIR, '..', 'view_planning', 'build', 'exploration_view_tests'))

# Print the resolved paths
print(f"scale_factor_file path: {scale_factor_file}")
print(f"object_model_path path: {object_model_path}")
print(f"cpp_simulator_path path: {cpp_simulator_path}")

# Check if the paths exist
print(f"Does '{scale_factor_file}' exist? {'Yes' if os.path.exists(scale_factor_file) else 'No'}")
print(f"Does '{object_model_path}' exist? {'Yes' if os.path.exists(object_model_path) else 'No'}")
print(f"Does '{cpp_simulator_path}' exist? {'Yes' if os.path.exists(cpp_simulator_path) else 'No'}")
