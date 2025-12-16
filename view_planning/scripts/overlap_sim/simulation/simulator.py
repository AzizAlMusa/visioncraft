# simulation/simulator.py
"""
Simulator
---------
Encapsulates the full potential-field simulation loop.

Responsibilities:
- Compute need map
- Compute attraction + repulsion fields
- Update viewpoint positions
- Handle automatic or manual viewpoint insertion
- Provide data for visualization
"""

try:
    import cupy as np
    GPU = True
except ImportError:
    import numpy as np
    GPU = False

from visibility.need_map import compute_need_map
from physics.fields import compute_fields, compute_fields_from_viewpoint
from physics.attraction import attraction_force_at_points
from physics.repulsion import compute_gaussian_repulsion
from simulation.optimizer import Optimizer
from simulation.insertion_manager import InsertionManager


class Simulator:
    def __init__(self, base_map, viewpoints,
                 k_attr=10.0, k_rep=0.0025,
                 sigma_rep=10.0, amp_rep=100.0,
                 fov_radius=20.0, beta_vis=20.0,
                 grid_size=100, optimizer=None):
        """
        Parameters
        ----------
        base_map : BaseMap
        viewpoints : Viewpoints
        optimizer : Optimizer, optional
        """

        self.base_map = base_map
        self.viewpoints = viewpoints
        self.grid_size = grid_size

        # --- simulation parameters ------------------------------------------
        self.k_attr = k_attr
        self.k_rep = k_rep
        self.sigma_rep = sigma_rep
        self.amp_rep = amp_rep
        self.fov_radius = fov_radius
        self.beta_vis = beta_vis
        self.optimizer = optimizer

        # Cached spatial grid
        self.X, self.Y = base_map.get_meshgrid()

        # Insertion manager
        self.insertion_manager = InsertionManager(
            mode="manual",
            insert_interval=50,
            velocity_thresh=0.02,
            stagnation_steps=10
        )

        self._step_count = 0

        print(f"[Simulator] Using {'CuPy GPU' if GPU else 'NumPy CPU'} backend")

    # --------------------------------------------------------
    def _compute_forces_with_need(self, need_map):
        """Internal helper used for stability/oscillation analysis.
        Always works in the simulator's backend (np = cupy or numpy).
        """
        # Normalize types to the simulator backend
        positions = np.asarray(self.viewpoints.positions)
        need_map_backend = np.asarray(need_map)

        # --- attraction -----------------------------------------------------
        F_attr = attraction_force_at_points(
            positions,          # (N, 2), simulator backend
            self.X,             # meshgrid already in simulator backend
            self.Y,
            need_map_backend,   # need map in same backend
            k_attr=self.k_attr,
            grid_size=self.grid_size
        )

        # --- repulsion ------------------------------------------------------
        N = positions.shape[0]
        F_rep = np.zeros_like(F_attr)

        for i in range(N):
            if i == 0:
                others = positions[1:]
            elif i == N - 1:
                others = positions[:-1]
            else:
                # np here is cupy or numpy; positions is already np.ndarray
                others = np.concatenate(
                    (positions[:i], positions[i + 1:]),
                    axis=0
                )

            # compute_gaussian_repulsion uses its own xp backend, so we
            # wrap the result back into the simulator backend with np.array
            fx, fy = compute_gaussian_repulsion(
                positions[i], others,
                sigma=self.sigma_rep,
                amp=self.amp_rep,
                grid_size=self.grid_size
            )
            F_rep[i, 0] = fx
            F_rep[i, 1] = fy

        forces = F_attr + self.k_rep * F_rep

        return {
            "forces_attr": F_attr,
            "forces_rep":  F_rep,
            "forces":      forces,
        }


    # --------------------------------------------------------
    def forces_given_positions(self, X, need_map):
        """Return total forces for arbitrary viewpoint positions."""
        _backup = self.viewpoints.positions
        try:
            self.viewpoints.positions = X
            res = self._compute_forces_with_need(need_map)
            return res["forces"]
        finally:
            self.viewpoints.positions = _backup

    # --------------------------------------------------------
    def step(self, i_active=None):
        """Advance the simulation by one iteration and compute all field quantities."""
        # --- (1) need map ----------------------------------------------------
        need_map = compute_need_map(
            self.X, self.Y, self.viewpoints.positions,
            fov_radius=self.fov_radius,
            beta=self.beta_vis,
            grid_size=self.grid_size
        )

        # --- (2) field grids -------------------------------------------------
        if i_active is None:
            potential, Fx_attr, Fy_attr, Fx_rep, Fy_rep, Fx_total, Fy_total = compute_fields(
                self.X, self.Y, self.viewpoints.positions, need_map,
                k_attr=self.k_attr, k_rep=self.k_rep,
                sigma_rep=self.sigma_rep, amp_rep=self.amp_rep,
                grid_size=self.grid_size
            )
        else:
            potential, Fx_attr, Fy_attr, Fx_rep, Fy_rep, Fx_total, Fy_total = compute_fields_from_viewpoint(
                self.X, self.Y, self.viewpoints.positions, i_active, need_map,
                k_attr=self.k_attr, k_rep=self.k_rep,
                sigma_rep=self.sigma_rep, amp_rep=self.amp_rep,
                grid_size=self.grid_size
            )

        # --- (3) attraction --------------------------------------------------
        F_attr = attraction_force_at_points(
            self.viewpoints.positions, self.X, self.Y, need_map,
            k_attr=self.k_attr, grid_size=self.grid_size
        )

        # --- (4) repulsion ---------------------------------------------------
        F_rep = np.zeros_like(F_attr)
        N = len(self.viewpoints.positions)
        for i in range(N):
            if i == 0:
                others = self.viewpoints.positions[1:]
            elif i == N - 1:
                others = self.viewpoints.positions[:-1]
            else:
                others = np.concatenate(
                    (self.viewpoints.positions[:i],
                     self.viewpoints.positions[i + 1:]),
                    axis=0
                )
            F_rep[i, :] = np.array(compute_gaussian_repulsion(
                self.viewpoints.positions[i], others,
                sigma=self.sigma_rep, amp=self.amp_rep,
                grid_size=self.grid_size
            ))

        forces = F_attr + self.k_rep * F_rep

        # --- (5) automatic insertion check ----------------------------------
        if self.insertion_manager.check_insertion(
            step=self._step_count,
            viewpoints=self.viewpoints,
            potential=potential
        ):
            new_pos = self.insertion_manager.select_best_insertion(
                potential, self.grid_size
            )
            self.viewpoints.add_viewpoint(new_pos)
            print(f"[Simulator] Inserted new viewpoint at {new_pos} (highest potential)")

        self._step_count += 1

        # --- (6) return GPU-native arrays directly --------------------------
        return {
            "need_map": need_map,
            "potential": potential,
            "Fx_attr": Fx_attr,
            "Fy_attr": Fy_attr,
            "Fx_rep": Fx_rep,
            "Fy_rep": Fy_rep,
            "Fx_total": Fx_total,
            "Fy_total": Fy_total,
            "forces_attr": F_attr,
            "forces_rep": F_rep,
            "forces": forces,
            "i_active": i_active,
        }
