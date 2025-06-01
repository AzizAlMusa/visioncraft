import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import argparse
import os

# --- Command line arguments ---
parser = argparse.ArgumentParser()
parser.add_argument("--potential_type", type=str, default="log")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--save_dir", type=str, default="./results")
args = parser.parse_args()

np.random.seed(args.seed)

# --- Simulation parameters ---
grid_size = 100
frames = 200
NEW_PARTICLES = 1
num_particles = 8
particles = 50 + np.random.rand(num_particles, 2) * 20
x = np.arange(grid_size)
y = np.arange(grid_size)
X, Y = np.meshgrid(x, y)
field_points = np.stack([X.ravel(), Y.ravel()], axis=-1)
epsilon = 1e-6

class Field:
    def __init__(self, grid_size, num_particles, fov_radius, use_wrapping=True):
        self.grid_size = grid_size
        self.fov_radius = fov_radius
        self.use_wrapping = use_wrapping
        self.visibility = np.zeros((grid_size, grid_size), dtype=np.float64)
        self.potential = np.zeros((grid_size, grid_size))
        self.field_points = field_points
        self.attractive_forces = np.zeros((grid_size, grid_size, num_particles, 2), dtype=np.float64)
        self.repulsive_forces = np.zeros((num_particles, num_particles, 2), dtype=np.float64)

    def compute_coverage_monte_carlo(self, num_samples=10000, threshold=0.0):
        random_points = np.random.rand(num_samples, 2) * self.grid_size
        diff = random_points[:, None, :] - particles[None, :, :]
        wrapped_diff = self.wrap_distance(diff)
        distances = np.linalg.norm(wrapped_diff, axis=-1) + epsilon
        visibility_values = (distances <= self.fov_radius).astype(float)
        sampled_visibility = np.clip(np.sum(visibility_values, axis=1), 0, 1)
        covered_samples = np.sum(sampled_visibility > threshold)
        return covered_samples / num_samples

    def compute_redundancy(self, num_samples=10000, threshold=0.25):
        random_points = np.random.rand(num_samples, 2) * self.grid_size
        diff = random_points[:, None, :] - particles[None, :, :]
        wrapped_diff = self.wrap_distance(diff)
        distances = np.linalg.norm(wrapped_diff, axis=-1)
        visibility_values = (distances <= self.fov_radius).astype(float)
        coverage_counts = np.sum(visibility_values, axis=1)
        redundant_points = np.sum(coverage_counts > 1)
        return redundant_points / num_samples

    def compute_overlap_affinity(self, num_samples=10000):
        random_points = np.random.rand(num_samples, 2) * self.grid_size
        diff = random_points[:, None, :] - particles[None, :, :]
        wrapped_diff = self.wrap_distance(diff)
        distances = np.linalg.norm(wrapped_diff, axis=-1)
        visibility_values = (distances <= self.fov_radius).astype(float)
        coverage_counts = np.sum(visibility_values, axis=1)
        valid = coverage_counts > 0
        return np.mean(coverage_counts[valid]) if np.any(valid) else 0.0

    def wrap_distance(self, diff):
        if not self.use_wrapping:
            return diff
        return np.where(np.abs(diff) > self.grid_size / 2,
                        -np.sign(diff) * (self.grid_size - np.abs(diff)), diff)

    def update_visibility(self, particles, beta=20.0):
        self.visibility.fill(0)
        diff = self.field_points[:, None, :] - particles[None, :, :]
        wrapped_diff = self.wrap_distance(diff)
        distances = np.linalg.norm(wrapped_diff, axis=-1) + epsilon
        s = 1.0 / (1.0 + np.exp(beta * (distances / self.fov_radius - 1.0)))
        self.visibility.ravel()[:] = np.clip(np.sum(s, axis=1), 0.0, 1.0)

    def compute_potential(self, particles, alpha=1.0):
        diff = self.field_points[:, None, :] - particles[None, :, :]
        wrapped_diff = self.wrap_distance(diff)
        distances = np.linalg.norm(wrapped_diff, axis=-1) + epsilon
        log_sum = np.log(distances).sum(axis=1)
        covered_factor = 1.0 - self.visibility.ravel()
        pot_values = alpha * covered_factor * log_sum
        self.potential = pot_values.reshape(self.grid_size, self.grid_size)

    def compute_attractive_force(self, particles, alpha=1.0):
        diff = self.field_points[:, None, :] - particles[None, :, :]
        wrapped_diff = self.wrap_distance(diff)
        distances = np.linalg.norm(wrapped_diff, axis=-1) + epsilon
        directions = wrapped_diff / distances[..., None]
        covered_factor = (1.0 - self.visibility.ravel())[:, None]
        coverage_need = alpha * covered_factor / distances
        coverage_need[distances < epsilon] = 0.0
        attractive_forces = coverage_need[..., None] * directions
        self.attractive_forces = attractive_forces.reshape(
            self.grid_size, self.grid_size, particles.shape[0], 2)
        return self.attractive_forces.sum(axis=(0, 1))


    def compute_repelling_force(self, particles, sigma=10, amplitude=100):
        pairwise_diff = particles[:, None, :] - particles[None, :, :]
        wrapped_pairwise_diff = self.wrap_distance(pairwise_diff)
        pairwise_distances = np.linalg.norm(wrapped_pairwise_diff, axis=-1) + epsilon
        repelling_forces = -(
            amplitude * wrapped_pairwise_diff *
            (-pairwise_distances[..., None] / sigma**2) *
            np.exp(-pairwise_distances**2 / (2 * sigma**2))[..., None]
        )
        self.repulsive_forces = repelling_forces
        return repelling_forces.sum(axis=1)

    def compute_force(self, particles, sigma=10, amplitude=100, k_attr=0.4, k_rep=0.0, alpha=1.0):
        total_attractive = self.compute_attractive_force(particles, alpha)
        total_repelling = self.compute_repelling_force(particles, sigma, amplitude)
        return k_attr * total_attractive + k_rep * total_repelling

field = Field(grid_size, num_particles, fov_radius=20)
coverage_data = []
redundancy_data = []
affinity_data = []

# Adam optimizer
beta1 = 0.9
beta2 = 0.999
learning_rate = 5
m = np.zeros_like(particles)
v = np.zeros_like(particles)
t = 0

for frame in range(frames):
    field.update_visibility(particles)
    coverage = field.compute_coverage_monte_carlo(num_samples=10000, threshold=0.25)
    redundancy = field.compute_redundancy(num_samples=10000, threshold=0.25)
    affinity = field.compute_overlap_affinity(num_samples=10000)
    coverage_data.append(coverage)
    redundancy_data.append(redundancy)
    affinity_data.append(affinity)
    print(f"[Frame {frame:03d}] Coverage: {coverage*100:.2f}% | Redundancy: {redundancy*100:.2f}% | Affinity: {affinity:.2f}")

    field.compute_potential(particles, alpha=1.0)
    total_forces = field.compute_force(particles, alpha=1.0)

    t += 1
    grad = total_forces
    m = beta1 * m + (1 - beta1) * grad
    v = beta2 * v + (1 - beta2) * (grad ** 2)
    m_hat = m / (1 - beta1**t)
    v_hat = v / (1 - beta2**t)
    particles += learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)
    particles %= grid_size

# Save output
os.makedirs(args.save_dir, exist_ok=True)
output_path = os.path.join(args.save_dir, f"{args.potential_type}_seed{args.seed}.npz")
np.savez_compressed(output_path,
    coverage_per_iter=np.array(coverage_data),
    redundancy_per_iter=np.array(redundancy_data),
    overlap_affinity_per_iter=np.array(affinity_data),
    potential_type=args.potential_type,
    seed=args.seed
)
print(f"[Saved] {output_path}")