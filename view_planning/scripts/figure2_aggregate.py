#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import os

# ---- Configuration ----
grid_size, fov_radius = 100, 20
epsilon = 1e-6
S = 50000
np.random.seed(0)

def wrap_distance(diff):
    half = grid_size / 2
    return np.where(np.abs(diff) > half, -np.sign(diff) * (grid_size - np.abs(diff)), diff)

def compute_visibility(viewpoint, field_points):
    diff = wrap_distance(field_points - viewpoint)
    dist = np.linalg.norm(diff, axis=-1) + epsilon
    beta = 20.0
    return 1.0 / (1.0 + np.exp(beta * (dist / fov_radius - 1.0)))

# ---- POTENTIAL FIELD ----
def run_potential_field():
    particles = np.random.rand(1, 2) * grid_size
    m = np.zeros_like(particles); v = np.zeros_like(particles)
    t_adam = 0
    window_size = 5
    recent_moves = []
    field_points = np.stack(np.meshgrid(np.arange(grid_size), np.arange(grid_size)), -1).reshape(-1, 2)
    visibility = np.zeros((grid_size, grid_size))

    def monte_carlo_coverage(particles):
        pts = np.random.rand(S, 2) * grid_size
        d = np.linalg.norm(wrap_distance(pts[:, None, :] - particles[None, :, :]), axis=-1)
        vis = d <= fov_radius
        return np.mean(vis.sum(axis=1) > 0.25)

    while True:
        diff = field_points[:, None, :] - particles[None, :, :]
        dist = np.linalg.norm(wrap_distance(diff), axis=-1) + epsilon
        vis = 1.0 / (1.0 + np.exp(20.0 * (dist / fov_radius - 1.0)))
        visibility.ravel()[:] = np.clip(vis.sum(axis=1), 0.0, 1.0)
        pot = (1.0 - visibility.ravel()) * np.log(dist).sum(axis=1)
        nbv = field_points[np.argmax(pot)]

        dirn = diff / dist[..., None]
        need = (1.0 - visibility.ravel())[:, None] / dist
        need[dist < epsilon] = 0.0
        attractive = (need[..., None] * dirn).reshape(grid_size, grid_size, particles.shape[0], 2).sum(axis=(0,1))

        pdiff = particles[:, None, :] - particles[None, :, :]
        pdist = np.linalg.norm(wrap_distance(pdiff), axis=-1) + epsilon
        sigma, amp = 10.0, 100.0
        rep = -(amp * pdiff * (-pdist[..., None] / sigma**2) * np.exp(-(pdist**2) / (2*sigma**2))[..., None]).sum(axis=1)
        forces = 10.0 * attractive + 0.25 * rep

        t_adam += 1
        m = 0.9 * m + 0.1 * forces
        v = 0.999 * v + 0.001 * (forces**2)
        step = 5 * (m/(1 - 0.9**t_adam)) / (np.sqrt(v/(1 - 0.999**t_adam)) + epsilon)
        particles = (particles + step) % grid_size

        move_mag = np.linalg.norm(step, axis=1).mean()
        recent_moves.append(move_mag)
        if len(recent_moves) > window_size:
            recent_moves.pop(0)

        if (len(particles) <= 2) or (len(recent_moves)==window_size and all(mv < 1.5 for mv in recent_moves)):
            particles = np.vstack([particles, nbv[None,:]])
            m = np.zeros_like(particles)
            v = np.zeros_like(particles)
            t_adam = 0
            recent_moves.clear()

        if monte_carlo_coverage(particles) >= 0.999 or len(particles) >= 100:
            break
    return particles

# ---- GREEDY ----
def run_greedy():
    field_points = np.stack(np.meshgrid(np.arange(grid_size), np.arange(grid_size)), -1).reshape(-1, 2)
    coverage = np.zeros(len(field_points))
    vps = []
    while True:
        if np.mean(coverage > 0.25) >= 1.0 or len(vps) >= 100:
            break
        best_gain = -1
        best_vp = None
        for cand in (field_points[np.random.choice(len(field_points), 100, replace=False)] + np.random.normal(0, fov_radius * 0.3, (100, 2))) % grid_size:
            vis = compute_visibility(cand, field_points)
            new_cov = np.maximum(coverage, vis)
            new_pts = np.sum((new_cov > 0.25) & (coverage <= 0.25))
            score = new_pts + 0.1 * new_cov[new_cov > 0.25].mean()
            if score > best_gain:
                best_gain = score
                best_vp = cand
        if best_gain > 0:
            vps.append(best_vp)
            coverage = np.maximum(coverage, compute_visibility(best_vp, field_points))
        else:
            break
    return np.array(vps)

# ---- GENETIC ----
def run_genetic():
    field_points = np.stack(np.meshgrid(np.arange(grid_size), np.arange(grid_size)), -1).reshape(-1, 2)
    candidates = (field_points[np.random.choice(len(field_points), 100, replace=False)] + np.random.normal(0, fov_radius * 0.3, (100, 2))) % grid_size
    A = np.array([compute_visibility(vp, field_points) for vp in candidates]).T
    pop = np.random.rand(50, 100)
    best_chrom, best_score, best_used = None, np.inf, []
    for _ in range(100):
        scores = []
        useds = []
        for chrom in pop:
            sorted_idx = np.argsort(chrom)
            covered = np.zeros(A.shape[0], dtype=bool)
            used = []
            for idx in sorted_idx:
                covered |= A[:, idx] >= 0.25
                used.append(idx)
                if covered.all(): break
            if not covered.all():
                scores.append(np.inf)
                useds.append([])
                continue
            redundancy = (A[:, used] >= 0.25).sum(axis=1)
            fitness = 1.0 * len(used) + 2.0 * np.std(redundancy)
            scores.append(fitness)
            useds.append(used)
        best_idx = np.argmin(scores)
        if scores[best_idx] < best_score:
            best_score = scores[best_idx]
            best_chrom = pop[best_idx]
            best_used = useds[best_idx]
        parents = pop[np.argsort(scores)[:25]]
        children = []
        for _ in range(25):
            p1, p2 = parents[np.random.randint(25)], parents[np.random.randint(25)]
            alpha = np.random.rand()
            child = alpha * p1 + (1 - alpha) * p2
            if np.random.rand() < 0.2:
                child[np.random.randint(100)] = np.random.rand()
            children.append(child)
        pop = np.vstack([parents, children])
    return candidates[best_used]

# ---- SIMULATED ANNEALING ----
def run_sa():
    vp = np.random.rand(15, 2) * grid_size
    field_points = np.stack(np.meshgrid(np.arange(grid_size), np.arange(grid_size)), -1).reshape(-1, 2)
    V = np.array([compute_visibility(p, field_points) for p in vp])
    cov_mask = np.maximum.reduce(V, axis=0)
    best_vp, best_cov = vp.copy(), np.mean(cov_mask > 0.25)
    for step in range(1000):
        T = 0.5 * (0.98 ** step)
        i = np.random.randint(15)
        new_vp = best_vp.copy()
        new_vp[i] = (new_vp[i] + np.random.normal(0, fov_radius * 0.3, 2)) % grid_size
        V_new = V.copy()
        V_new[i] = compute_visibility(new_vp[i], field_points)
        cov_new = np.maximum.reduce(V_new, axis=0)
        cov_val = np.mean(cov_new > 0.25)
        penalty = 10.0 * (1.0 - cov_val)**2 + 5.0 * np.std((V_new >= 0.25).sum(axis=1))
        best_penalty = 10.0 * (1.0 - best_cov)**2 + 5.0 * np.std((V >= 0.25).sum(axis=1))
        if penalty < best_penalty or np.random.rand() < np.exp(-(penalty - best_penalty) / (T + epsilon)):
            V = V_new
            best_vp = new_vp
            best_cov = cov_val
        if best_cov >= 1.0:
            break
    return best_vp

# ---- Plotting ----
def plot_final(ax, pts, title):
    field_points = np.stack(np.meshgrid(np.arange(grid_size), np.arange(grid_size)), -1).reshape(-1, 2)
    vis_mask = np.zeros(len(field_points), dtype=bool)
    for p in pts:
        dist = np.linalg.norm(wrap_distance(field_points - p), axis=1)
        vis_mask |= dist <= fov_radius
    visibility = vis_mask.astype(np.float32)
    vis_field = np.zeros((grid_size * grid_size,))
    vis_field[:len(visibility)] = visibility
    vis_img = 1 - vis_field.reshape(grid_size, grid_size)
    ax.imshow(vis_img, cmap='viridis', origin='lower',
              extent=[0, grid_size, 0, grid_size], vmin=0, vmax=1)
    for p in pts:
        for dx in (-grid_size, 0, grid_size):
            for dy in (-grid_size, 0, grid_size):
                ax.add_patch(plt.Circle((p[0]+dx, p[1]+dy), fov_radius,
                              edgecolor='white', facecolor='none', alpha=0.2, lw=1))
    ax.scatter(pts[:, 0], pts[:, 1], c='deepskyblue', s=30,
               edgecolors='white', linewidths=0.6)
    ax.set_xlim(0, grid_size); ax.set_ylim(0, grid_size)
    ax.set_title(title, fontsize=12); ax.axis('off')

# ---- Run All and Plot ----
fig, axs = plt.subplots(2, 2, figsize=(12, 12))
axs = axs.ravel()  # 🔧 Flatten the 2x2 grid to a 1D list

plot_final(axs[0], run_potential_field(), "POTENTIAL FIELD")
plot_final(axs[1], run_greedy(), "GREEDY")
plot_final(axs[2], run_genetic(), "GENETIC")
plot_final(axs[3], run_sa(), "SIMULATED ANNEALING")

plt.tight_layout()
plt.savefig("./results2/final_state_comparison_from_scratch.png", dpi=300)
plt.show()

 
