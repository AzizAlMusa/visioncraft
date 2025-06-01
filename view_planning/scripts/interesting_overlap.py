import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.gridspec import GridSpec

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
grid_size   = 100                 # side of the square workspace
frames      = 600                 # animation length
num_particles  = 10                # starting viewpoints
fov_radius  = 20                  # field-of-view radius
NEW_PARTICLES = 1                 # how many to add / remove
ADD_FRAMES    = []                # e.g. [50,100,150 ...]
REMOVE_FRAMES = []                # e.g. [400]

np.random.seed(0)                 # reproducible demo

# ---------------------------------------------------------------------------
# Helper data: grid + "interesting" cells mask
# ---------------------------------------------------------------------------
x = np.arange(grid_size)
y = np.arange(grid_size)
X, Y = np.meshgrid(x, y)                              # shape (g,g)
field_points = np.stack([X.ravel(), Y.ravel()], -1)   # (g²,2)

# --- Create blob-shaped interesting regions in a Z pattern with Gaussian softening -----
# First create a binary mask for the core of the interesting regions
binary_mask = np.zeros((grid_size, grid_size), dtype=bool)

# Define blob centers in a Z pattern across the grid
blob_centers = [
    # Top row of the Z (left to right)
    (15, 80),
   
    
    # Diagonal of the Z (from top-right to bottom-left)
    (65, 65),

    

]

# Vary the blob sizes slightly for visual interest
blob_radii = [
    15, 15
]

# Create the binary core mask
for (cx, cy), radius in zip(blob_centers, blob_radii):
    blob = ((X - cx)**2 + (Y - cy)**2) <= radius**2
    binary_mask = np.logical_or(binary_mask, blob)

# Now create a continuous gradient mask with Gaussian falloff
interesting_mask = np.zeros((grid_size, grid_size))

# Add each blob with Gaussian falloff
for (cx, cy), radius in zip(blob_centers, blob_radii):
    # Calculate squared distance from center
    sq_dist = ((X - cx)**2 + (Y - cy)**2)
    # Gaussian falloff - sigma controls the softness of the falloff
    sigma = radius * 0.5  # Controls softening
    gaussian = np.exp(-sq_dist / (2 * sigma**2))
    interesting_mask = np.maximum(interesting_mask, gaussian)

# Normalize to 0-1 range
interesting_mask = interesting_mask / np.max(interesting_mask)
# ---------------------------------------------------------------------------

epsilon = 1e-6                                        # tiny

# ---------------------------------------------------------------------------
# Field class
# ---------------------------------------------------------------------------
class Field:
    def __init__(self, grid_size, num_particles, fov_radius,
                 interesting_mask, use_wrapping=True,
                 normal_scheme=None, interesting_scheme=None,
                 visibility_sharpness=12.0):
        self.grid_size = grid_size
        self.fov_radius = fov_radius
        self.use_wrapping = use_wrapping
        self.interesting_mask = interesting_mask  # Now a continuous mask (0-1)

        self.normal_scheme = normal_scheme #or create_scheme(0.0, 1.0, 0.5, 0.2)
        self.interesting_scheme = interesting_scheme #or create_scheme(0.2, 0.0, 1.0, 0.4)
        self.visibility_sharpness = visibility_sharpness

        self.visibility = np.zeros((grid_size, grid_size))
        self.overlap_count = np.zeros((grid_size, grid_size), dtype=int)
        self.optimal_overlap = np.zeros((grid_size, grid_size))
        self.potential = np.zeros((grid_size, grid_size))
        self.field_points = field_points

        self.attractive_forces = np.zeros((grid_size, grid_size, num_particles, 2))
        self.repulsive_forces = np.zeros((num_particles, num_particles, 2))

        self.theoretical_max_coverage = min(1.0, num_particles * np.pi * fov_radius**2 / grid_size**2)

    def update_visibility_schemes(self, normal_scheme=None, interesting_scheme=None, sharpness=None):
        if normal_scheme is not None:
            self.normal_scheme = normal_scheme
        if interesting_scheme is not None:
            self.interesting_scheme = interesting_scheme
        if sharpness is not None:
            self.visibility_sharpness = sharpness

    def wrap_distance(self, diff):
        if not self.use_wrapping:
            return diff
        g2 = self.grid_size / 2
        return np.where(np.abs(diff) > g2, -np.sign(diff) * (self.grid_size - np.abs(diff)), diff)

    def update_visibility(self, particles):
        diff = self.field_points[:, None, :] - particles[None, :, :]
        wrapped = self.wrap_distance(diff)
        dists = np.linalg.norm(wrapped, axis=-1)
        decay = 0.3
        contrib = 1.0 / (1.0 + np.exp((dists - self.fov_radius) * decay))
        eff_cov = np.sum(contrib, axis=1)

        q_int = np.zeros_like(eff_cov)
        for i, (lvl, val) in enumerate(self.interesting_scheme):
            if i == len(self.interesting_scheme) - 1:
                w = 1.0 / (1.0 + np.exp(-self.visibility_sharpness * (eff_cov - lvl)))
            else:
                nxt = self.interesting_scheme[i+1][0]
                rise = 1.0 / (1.0 + np.exp(-self.visibility_sharpness * (eff_cov - lvl)))
                fall = 1.0 / (1.0 + np.exp(-self.visibility_sharpness * (eff_cov - nxt)))
                w = rise * (1.0 - fall)
            q_int += val * w

        q_norm = np.zeros_like(eff_cov)
        for i, (lvl, val) in enumerate(self.normal_scheme):
            if i == len(self.normal_scheme) - 1:
                w = 1.0 / (1.0 + np.exp(-self.visibility_sharpness * (eff_cov - lvl)))
            else:
                nxt = self.normal_scheme[i+1][0]
                rise = 1.0 / (1.0 + np.exp(-self.visibility_sharpness * (eff_cov - lvl)))
                fall = 1.0 / (1.0 + np.exp(-self.visibility_sharpness * (eff_cov - nxt)))
                w = rise * (1.0 - fall)
            q_norm += val * w

        # Apply weighting based on the continuous interesting_mask
        mask_flat = self.interesting_mask.ravel()
        # Blend between normal and interesting based on mask value (0-1)
        quality = mask_flat * q_int + (1 - mask_flat) * q_norm
        self.visibility = quality.reshape(self.grid_size, self.grid_size)

        self.update_binary_overlap(particles)

    def update_binary_overlap(self, particles):
        diff = self.field_points[:, None, :] - particles[None, :, :]
        d = np.linalg.norm(self.wrap_distance(diff), axis=-1)
        covered = (d <= self.fov_radius)
        overlap = np.sum(covered, axis=1)
        self.overlap_count = overlap.reshape(self.grid_size, self.grid_size)
        self.optimal_overlap = (overlap == 2).astype(float).reshape(self.grid_size, self.grid_size)

    def compute_coverage_monte_carlo(self, particles, num_samples=10000, threshold=0.0):
        rnd = np.random.rand(num_samples, 2) * self.grid_size
        diff = rnd[:, None, :] - particles[None, :, :]
        d = np.linalg.norm(self.wrap_distance(diff), axis=-1)
        covered = np.any(d <= self.fov_radius, axis=1)
        coverage_fraction = covered.mean()

        overlap = np.sum(d <= self.fov_radius, axis=1)
        covered_overlap = overlap[covered]

        if covered_overlap.size:
            single = (covered_overlap == 1).mean()
            double = (covered_overlap == 2).mean()
            excess = (covered_overlap > 2).mean()
            optimal_ratio = ((covered_overlap==1)|(covered_overlap==2)).sum()/covered_overlap.size
            total_units = overlap.sum()
            redundant = total_units - covered.size
            red_ratio = redundant / total_units if total_units else 0.
        else:
            single=double=excess=optimal_ratio=red_ratio=0.

        n = particles.shape[0]
        theor = min(1.0, n*np.pi*self.fov_radius**2/self.grid_size**2)
        self.theoretical_max_coverage = theor

        return dict(coverage=coverage_fraction,
                    single_coverage=single, double_coverage=double,
                    excess_coverage=excess, optimal_ratio=optimal_ratio,
                    redundancy_ratio=red_ratio, theoretical_max=theor)

    def compute_potential(self, particles, alpha=1.0):
        diff = self.field_points[:, None, :] - particles[None, :, :]
        wrapped = self.wrap_distance(diff)
        distances = np.linalg.norm(wrapped, axis=-1) + epsilon
        log_sum = np.log(distances).sum(axis=1)
        coverage_need = 1.0 - self.visibility.ravel()
        self.potential = (alpha * coverage_need * log_sum).reshape(self.grid_size, self.grid_size)

    def compute_attractive_force(self, particles, alpha=1.0):
        diff = self.field_points[:, None, :] - particles[None, :, :]
        wrapped = self.wrap_distance(diff)
        distances = np.linalg.norm(wrapped, axis=-1) + epsilon
        directions = wrapped / distances[..., None]

        coverage_need = (1.0 - self.visibility.ravel())[:, None]
        force_mag = alpha * coverage_need / distances
        force_mag[distances < epsilon] = 0.0

        attractive = force_mag[..., None] * directions
        self.attractive_forces = attractive.reshape(self.grid_size, self.grid_size, particles.shape[0], 2)

        return self.attractive_forces.sum(axis=(0, 1))

    def compute_repelling_force(self, particles, sigma=10, amplitude=100):
        pdiff = self.wrap_distance(particles[:, None, :] - particles[None, :, :])
        dist = np.linalg.norm(pdiff, axis=-1) + epsilon
        rep = -(amplitude * pdiff *
                (-dist[..., None]/sigma**2) *
                np.exp(-dist**2/(2*sigma**2))[..., None])
        self.repulsive_forces = rep
        return rep.sum(axis=1)

    def compute_force(self, particles, **kw):
        k_attr = kw.get('k_attr', 4.0)
        k_rep = kw.get('k_rep', 0.0)
        total_attr = self.compute_attractive_force(particles, kw.get('alpha', 1.0))
        total_rep = self.compute_repelling_force(particles, kw.get('sigma', 10), kw.get('amplitude', 100))
        return k_attr * total_attr + k_rep * total_rep

# ---------------------------------------------------------------------------
# Helper function to visualize current schemes
# ---------------------------------------------------------------------------
def plot_visibility_schemes(field, ax=None):
    """
    Visualize the current normal and interesting schemes as curves.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    
    # Generate a range of coverage values
    x = np.linspace(0, 4, 1000)
    
    # Calculate quality values for normal and interesting schemes
    y_normal = np.zeros_like(x)
    for i, (threshold, value) in enumerate(field.normal_scheme):
        if i == len(field.normal_scheme) - 1:
            w = 1.0 / (1.0 + np.exp(-field.visibility_sharpness * (x - threshold)))
        else:
            next_threshold = field.normal_scheme[i+1][0]
            rise = 1.0 / (1.0 + np.exp(-field.visibility_sharpness * (x - threshold)))
            fall = 1.0 / (1.0 + np.exp(-field.visibility_sharpness * (x - next_threshold)))
            w = rise * (1.0 - fall)
        y_normal += value * w
    
    y_interesting = np.zeros_like(x)
    for i, (threshold, value) in enumerate(field.interesting_scheme):
        if i == len(field.interesting_scheme) - 1:
            w = 1.0 / (1.0 + np.exp(-field.visibility_sharpness * (x - threshold)))
        else:
            next_threshold = field.interesting_scheme[i+1][0]
            rise = 1.0 / (1.0 + np.exp(-field.visibility_sharpness * (x - threshold)))
            fall = 1.0 / (1.0 + np.exp(-field.visibility_sharpness * (x - next_threshold)))
            w = rise * (1.0 - fall)
        y_interesting += value * w
    
    # Plot the curves
    ax.plot(x, y_normal, 'b-', label='Normal Cells')
    ax.plot(x, y_interesting, 'r-', label='Interesting Cells')
    
    # Mark the thresholds
    for threshold, value in field.normal_scheme:
        ax.axvline(x=threshold, color='b', linestyle='--', alpha=0.3)
    
    for threshold, value in field.interesting_scheme:
        ax.axvline(x=threshold, color='r', linestyle='--', alpha=0.3)
    
    # Add grid, labels, and legend
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('Coverage')
    ax.set_ylabel('Quality')
    ax.set_title('Visibility Quality vs. Coverage')
    ax.legend()
    ax.set_xlim(0, 4)
    ax.set_ylim(0, 1.1)
    return ax

# ---------------------------------------------------------------------------
# Initial state with custom schemes for demonstration
# ---------------------------------------------------------------------------
# Define your schemes - these can be customized
normal_scheme = [
    (0.0, 0.0),  # 0 coverage -> 0.0 quality
    (1.0, 1.0),  # 1 coverage -> 1.0 quality 
    (2.0, 1.0),  # 2 coverage -> 0.5 quality
    (3.0, 1.0)   # 3+ coverage -> 0.2 quality
]

interesting_scheme = [
    (0.0, 0.2),  # 0 coverage -> 0.2 quality
    (1.0, 0.0),  # 1 coverage -> 0.0 quality
    (2.0, 1.0),  # 2 coverage -> 1.0 quality
    (3.0, 1.0)   # 3+ coverage -> 0.4 quality
]

particles = 50 + np.random.rand(num_particles, 2)       # roughly center

# Create field with custom schemes
field = Field(grid_size, num_particles, fov_radius,
              interesting_mask,
              normal_scheme=normal_scheme,
              interesting_scheme=interesting_scheme,
              visibility_sharpness=12.0)

# ---------------------------------------------------------------------------
# Matplotlib figure
# ---------------------------------------------------------------------------
fig = plt.figure(figsize=(15,8), dpi=150)
gs  = GridSpec(2,2, width_ratios=[1,1], height_ratios=[3,1])
ax_main   = fig.add_subplot(gs[0,0])
ax_line   = fig.add_subplot(gs[0,1])
ax_overlap= fig.add_subplot(gs[1,:])

plt.subplots_adjust(wspace=0.3, hspace=0.3)

# --- main plot -------------------------------------------------------------
ax_main.set_xlim(0,grid_size-1); ax_main.set_ylim(0,grid_size-1)
ax_main.set_aspect('equal'); ax_main.set_title('Field Potential & Coverage')
# show interesting regions with continuous gradient
im = ax_main.imshow(interesting_mask, cmap='Reds', alpha=0.5,
                   origin='lower', extent=[0,grid_size,0,grid_size], zorder=0)

# --- right-upper plot (coverage) ------------------------------------------
ax_line.set_xlim(0,frames); ax_line.set_ylim(0,1)
ax_line.set_title('Coverage Over Time'); ax_line.set_xlabel('Time Step')
ax_line.set_ylabel('Coverage')

# --- bottom plot (overlap metrics) ----------------------------------------
ax_overlap.set_xlim(0,frames); ax_overlap.set_ylim(0,1)
ax_overlap.set_title('Overlap Metrics'); ax_overlap.set_xlabel('Time Step')
ax_overlap.set_ylabel('Ratio'); ax_overlap.legend(loc='upper right')

# ---------------------------------------------------------------------------
# Artists that will be updated
# ---------------------------------------------------------------------------
scatter = ax_main.scatter(particles[:,0], particles[:,1],
                          c='deepskyblue', s=50, zorder=5)
contour = ax_main.contourf(X, Y, np.zeros_like(X),
                           levels=100, cmap='viridis',
                           alpha=0.9, origin='lower')
quiver  = None
coverage_line,      = ax_line.plot([], [], color='#3d03fc', label='Coverage')
max_coverage_line,  = ax_line.plot([], [], 'r--', label='Max Theoretical')
single_line,     = ax_overlap.plot([], [], 'g-',  label='Single')
double_line,     = ax_overlap.plot([], [], 'b-',  label='Double')
excess_line,     = ax_overlap.plot([], [], 'r-',  label='Excess')
redundancy_line, = ax_overlap.plot([], [], 'k--', label='Redundancy')


for ax in (ax_overlap,): ax.legend(loc='upper right', ncol=4, fontsize=8)
for ax in (ax_line,):    ax.legend(loc='upper right', fontsize=8)

# ---------------------------------------------------------------------------
# Optimiser state
# ---------------------------------------------------------------------------
beta1=0.9; beta2=0.999; lr=5; adam_eps=1e-8
m = np.zeros_like(particles); v = np.zeros_like(particles); t_adam=0

# stats history
cov_h=[]; maxcov_h=[]; s1_h=[]; s2_h=[]; exc_h=[]; red_h=[]
cbar = None; current_text=num_text=optimal_text=None
circles=[]

# ---------------------------------------------------------------------------
# Animation callback
# ---------------------------------------------------------------------------
def update(frame):
    global particles,contour,quiver,cbar,m,v,t_adam,current_text,num_text,optimal_text,circles

    # ---- dynamic scheme changes (can be uncommented for experimentation) ---
    # if frame == 100:
    #     print("Changing scheme at frame 100")
    #     field.update_visibility_schemes(
    #         normal_scheme=[(0.0, 0.0), (1.0, 1.0), (2.0, 0.5)]
    #     )
    # elif frame == 300:
    #     print("Changing scheme at frame 300")
    #     field.update_visibility_schemes(
    #         normal_scheme=[(1.0, 1.0)],
    #         interesting_scheme=[(0.0, 0.2), (1.0, 0.0), (2.0, 1.0)]
    #     )
    
    # ---- dynamic add / remove --------------------------------------------
    if frame in ADD_FRAMES:
        new = np.random.rand(NEW_PARTICLES,2)*grid_size
        particles = np.vstack([particles,new])
        m = np.vstack([m,np.zeros_like(new)])
        v = np.vstack([v,np.zeros_like(new)])
        field.repulsive_forces = np.zeros(
              (particles.shape[0], particles.shape[0], 2))
    if frame in REMOVE_FRAMES and particles.shape[0]>NEW_PARTICLES:
        particles = particles[:-NEW_PARTICLES]
        m=v=m[:particles.shape[0]], v[:particles.shape[0]]
        field.repulsive_forces = np.zeros(
              (particles.shape[0], particles.shape[0], 2))

    # ---- physics ---------------------------------------------------------
    field.update_visibility(particles)
    metrics = field.compute_coverage_monte_carlo(particles,10000)
    field.compute_potential(particles)
    forces = field.compute_force(particles,k_rep=0.05)

    # Adam step
    t_adam += 1
    m = beta1*m + (1-beta1)*forces
    v = beta2*v + (1-beta2)*(forces**2)
    m_hat = m / (1-beta1**t_adam)
    v_hat = v / (1-beta2**t_adam)
    particles = (particles + lr * m_hat / (np.sqrt(v_hat)+adam_eps)) % grid_size

    # ---- update plots ----------------------------------------------------
    # contour
    for c in contour.collections: c.remove()
    contour = ax_main.contourf(X,Y,field.potential,
                               levels=100,cmap='viridis',
                               alpha=1.0,origin='lower')
    if cbar: cbar.remove()
    div = make_axes_locatable(ax_main); cax=div.append_axes('right',size='5%',pad=0.05)
    cbar = plt.colorbar(contour,cax=cax)

    # quiver (first particle's attractive field)
    step=5
    xq,yq = X[::step,::step], Y[::step,::step]
    if quiver: quiver.remove()
    u_attr = field.attractive_forces[::step,::step,0,0].flatten()
    v_attr = field.attractive_forces[::step,::step,0,1].flatten()
    quiver = ax_main.quiver(xq,yq,u_attr,v_attr,
                            color='#ff1493',scale=1000,width=0.002,pivot='middle',zorder=99)

    # particles scatter + circles
    scatter.set_offsets(particles)
    for cir in circles:
        try: cir.remove()
        except: pass
    circles=[]
    
    # Generate more wrapping points to show circles that wrap around edges
    for p in particles:
        # Create circles for the particle and its 8 wrapping positions
        for dx in (-grid_size, 0, grid_size):
            for dy in (-grid_size, 0, grid_size):
                # Create circle for each wrapping position
                cir = plt.Circle((p[0]+dx, p[1]+dy), fov_radius, fill=False,
                                 color='white', lw=0.7, alpha=0.4, zorder=4)
                ax_main.add_patch(cir)
                circles.append(cir)

    # time-series data
    cov_h.append(metrics['coverage']);      maxcov_h.append(metrics['theoretical_max'])
    s1_h.append(metrics['single_coverage']);s2_h.append(metrics['double_coverage'])
    exc_h.append(metrics['excess_coverage']);red_h.append(metrics['redundancy_ratio'])

    coverage_line.set_data(range(len(cov_h)),cov_h)
    max_coverage_line.set_data(range(len(maxcov_h)),maxcov_h)
    single_line.set_data(range(len(s1_h)),s1_h)
    double_line.set_data(range(len(s2_h)),s2_h)
    excess_line.set_data(range(len(exc_h)),exc_h)
    redundancy_line.set_data(range(len(red_h)),red_h)

    # marker on last point
    coverage_line.set_marker('o'); coverage_line.set_markersize(4)
    coverage_line.set_markevery([len(cov_h)-1])

    # texts
    for txt in (current_text,num_text,optimal_text):
        if txt: txt.remove()
    ratio = (metrics['coverage']/metrics['theoretical_max']
             if metrics['theoretical_max']>0 else 0)
    current_text = ax_line.text(0.95,0.95,
        f"Coverage: {metrics['coverage']*100:.2f}%\n"
        f"Max: {metrics['theoretical_max']*100:.2f}%\n"
        f"Ratio: {ratio*100:.1f}%",
        transform=ax_line.transAxes, ha='right',va='top',fontsize=8)
    num_text = ax_main.text(0.02,0.98,f"Viewpoints: {particles.shape[0]}",
                            transform=ax_main.transAxes,fontsize=8,ha='left',va='top')
    optimal_text = ax_overlap.text(0.95,0.95,
        f"Optimal: {metrics['optimal_ratio']*100:.2f}%\n"
        f"Redundancy: {metrics['redundancy_ratio']*100:.2f}%",
        transform=ax_overlap.transAxes, ha='right',va='top',fontsize=8)

    # dynamic x-axis length
    for ax in (ax_line,ax_overlap):
        ax.set_xlim(0,max(len(cov_h),frames))
    return (scatter,contour,quiver,coverage_line,max_coverage_line,
            single_line,double_line,excess_line,redundancy_line)

# ---------------------------------------------------------------------------
# Run animation
# ---------------------------------------------------------------------------
ani = animation.FuncAnimation(fig, update, frames=frames,
                              interval=100, blit=False)
plt.tight_layout()
plt.show()