# postprocessing/perturbation_experiment.py

import numpy as np
import matplotlib.pyplot as plt

from config.sim_config import SimConfig
from map.base_map import BaseMap
from entities.viewpoint import Viewpoints
from simulation.simulator import Simulator
from simulation.optimizer import Optimizer
from simulation.stability_logger import _to_numpy, _torus_delta  # or reimplement delta here


def run_perturbation_experiment(
    N=12,
    steps_before=200,
    steps_after=200,
    perturb_idx=0,
    dx=1.0,
    dy=0.0,
):
    """
    Run a closed-loop experiment:
      - simulate until near steady-state
      - apply a finite kick to viewpoint `perturb_idx`
      - measure RMS distance back to the pre-kick configuration
    """
    cfg = SimConfig(
        num_viewpoints=N,
        max_steps=steps_before + steps_after,
    )

    base_map = BaseMap(grid_size=cfg.grid_size)
    viewpoints = Viewpoints(num=cfg.num_viewpoints,
                            grid_size=cfg.grid_size,
                            seed=cfg.seed)

    opt = Optimizer(
        name=cfg.optimizer,
        step_size=cfg.step_size,
        beta1=cfg.adam_beta1,
        beta2=cfg.adam_beta2,
        eps=cfg.adam_eps,
        gain=cfg.adam_gain,
    )

    sim = Simulator(
        base_map=base_map,
        viewpoints=viewpoints,
        k_attr=cfg.k_attr,
        k_rep=cfg.k_rep,
        sigma_rep=cfg.sigma_rep,
        amp_rep=cfg.amp_rep,
        fov_radius=cfg.fov_radius,
        beta_vis=cfg.beta_vis,
        grid_size=cfg.grid_size,
        optimizer=opt,
    )

    ref_pos = None
    distances = []
    rel_steps = []

    for step in range(steps_before + steps_after):
        # normal simulation step (no viewer)
        results = sim.step(i_active=None)

        pos_np = _to_numpy(viewpoints.positions)

        if step == steps_before - 1:
            # freeze reference configuration just before the kick
            ref_pos = pos_np.copy()

        if step == steps_before:
            # apply the kick
            pos_np[perturb_idx, 0] = (pos_np[perturb_idx, 0] + dx) % cfg.grid_size
            pos_np[perturb_idx, 1] = (pos_np[perturb_idx, 1] + dy) % cfg.grid_size

            # write back with correct backend
            try:
                import cupy as cp
                viewpoints.positions = cp.asarray(pos_np)
            except ImportError:
                viewpoints.positions = pos_np

        if step >= steps_before and ref_pos is not None:
            delta = _torus_delta(pos_np, ref_pos, cfg.grid_size)
            norms = np.linalg.norm(delta, axis=1)
            dist_rms = float(np.sqrt((norms ** 2).mean()))
            distances.append(dist_rms)
            rel_steps.append(step - steps_before)

    # Plot the relaxation envelope
    plt.figure()
    plt.plot(rel_steps, distances)
    plt.xlabel("steps after perturbation")
    plt.ylabel("RMS distance to pre-kick configuration")
    plt.title(f"N={N}, kick viewpoint {perturb_idx}")
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    run_perturbation_experiment(N=12, steps_before=200, steps_after=200,
                                perturb_idx=0, dx=1.0, dy=0.0)
