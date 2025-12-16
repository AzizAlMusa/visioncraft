# visualization/viewer_dual.py
"""
Dual-panel viewer for overlap_sim
---------------------------------
Left : Potential field
Right: Vector field (attraction, repulsion, or total)

Keyboard controls
    p  - Pause / resume
    q  - Quit
    a  - Attraction only
    r  - Repulsion only
    t  - Total (default)
    b  - Both attraction + repulsion
    n  - Cycle selected viewpoint (for highlighting)

Author: Abdulaziz (overlap_sim)
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle


class ViewerDual:
    def __init__(self, field, enable=True, target_fps=20, cmap="plasma",
                 show_vectors=True, i_selected=0):
        self.enabled = bool(enable)
        if not self.enabled:
            return

        self.field = field
        self.grid_size = field.grid_size
        self.fov_radius = getattr(field, "fov_radius", 20.0)
        self.target_dt = 1.0 / target_fps
        self.cmap = cmap
        self.show_vectors = show_vectors
        self.i_selected = i_selected

        plt.style.use("dark_background")
        plt.ion()

        # -------------------------------------------------------
        # Layout
        # -------------------------------------------------------
        if show_vectors:
            self.fig, (self.ax_left, self.ax_right) = plt.subplots(
                1, 2, figsize=(14, 7), constrained_layout=True)
        else:
            self.fig, self.ax_left = plt.subplots(figsize=(7, 7))
            self.ax_right = None

        # -------------------------------------------------------
        # Left panel
        # -------------------------------------------------------
        self.ax_left.set_xlim(0, self.grid_size)
        self.ax_left.set_ylim(0, self.grid_size)
        self.ax_left.set_aspect("equal", "box")
        self.ax_left.set_title("Potential Field", fontsize=12)
        self.ax_left.set_facecolor("#111111")

        self.im = self.ax_left.imshow(
            np.zeros((self.grid_size, self.grid_size)),
            cmap=self.cmap, origin="lower",
            extent=[0, self.grid_size, 0, self.grid_size])
        self.cbar = self.fig.colorbar(self.im, ax=self.ax_left,
                                      fraction=0.046, pad=0.04)
        self.cbar.set_label("Potential", fontsize=10)

        self.scat = self.ax_left.scatter([], [], s=60, c="#FFD166",
                                         edgecolors="black", linewidths=0.8, zorder=4)
        self.fov_patches = []
        self.text = self.ax_left.text(3, self.grid_size - 4, "",
                                      color="white", fontsize=10,
                                      ha="left", va="top")

        # -------------------------------------------------------
        # Right panel (vector fields + viewpoints)
        # -------------------------------------------------------
        self.mode = "total"
        if self.ax_right is not None:
            self.ax_right.set_xlim(0, self.grid_size)
            self.ax_right.set_ylim(0, self.grid_size)
            self.ax_right.set_aspect("equal", "box")
            self.ax_right.set_title("Vector Field", fontsize=12)
            self.ax_right.set_facecolor("#111111")

            stride = max(1, self.grid_size // 20)
            Xs = np.arange(0, self.grid_size, stride)
            Ys = np.arange(0, self.grid_size, stride)
            Xg, Yg = np.meshgrid(Xs, Ys)
            U0 = np.zeros_like(Xg)
            V0 = np.zeros_like(Yg)

            # Aqua-inspired palette
            self.quiv_attr = self.ax_right.quiver(Xg, Yg, U0, V0,
                                                  color="#00E0FF", scale=10)
            self.quiv_rep  = self.ax_right.quiver(Xg, Yg, U0, V0,
                                                  color="#FF00C8", scale=10)
            self.quiv_tot  = self.ax_right.quiver(Xg, Yg, U0, V0,
                                                  color="#00FFAA", scale=10)

            # viewpoint overlay on right panel
            self.scat_right = self.ax_right.scatter([], [], s=60,
                                                    c="#FFD166",
                                                    edgecolors="black",
                                                    linewidths=0.8, zorder=4)
            self.highlight = self.ax_right.scatter([], [], s=90,
                                                   c="#FF5500", marker="*",
                                                   edgecolors="white",
                                                   linewidths=1.2, zorder=5)

            self._quiv_stride = stride
            self._quiv_X, self._quiv_Y = Xg, Yg
            self._set_quiver_visibility()

        # -------------------------------------------------------
        # Interaction
        # -------------------------------------------------------
        self.paused = False
        self.quit = False
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)

    # =========================================================
    # Keyboard
    # =========================================================
    def _on_key(self, event):
        if event.key == "p":
            self.paused = not self.paused
            print(f"[ViewerDual] Paused={self.paused}")
        elif event.key == "q":
            self.quit = True
            plt.close(self.fig)
            print("[ViewerDual] Quit requested.")
        elif event.key == "a":
            self.mode = "attr"; self._set_quiver_visibility()
            print("[ViewerDual] Showing attraction field")
        elif event.key == "r":
            self.mode = "rep"; self._set_quiver_visibility()
            print("[ViewerDual] Showing repulsion field")
        elif event.key == "t":
            self.mode = "total"; self._set_quiver_visibility()
            print("[ViewerDual] Showing total field")
        elif event.key == "b":
            self.mode = "both"; self._set_quiver_visibility()
            print("[ViewerDual] Showing both attraction + repulsion")
        elif event.key == "n":
            self.i_selected = (self.i_selected + 1) % self.scat_right.get_offsets().shape[0]
            print(f"[ViewerDual] Selected viewpoint index: {self.i_selected}")

    # ---------------------------------------------------------
    def _set_quiver_visibility(self):
        if self.ax_right is None:
            return
        self.quiv_attr.set_visible(self.mode in ["attr", "both"])
        self.quiv_rep.set_visible(self.mode in ["rep", "both"])
        self.quiv_tot.set_visible(self.mode == "total")
    
    # =========================================================
    # Field-of-view circles
    # =========================================================
    def _update_fov(self, particles):
        for c in self.fov_patches:
            c.remove()
        self.fov_patches.clear()
        g, r = self.grid_size, self.fov_radius
        offsets = [(0, 0), (-g, 0), (g, 0), (0, -g), (0, g),
                   (-g, -g), (-g, g), (g, -g), (g, g)]
        for p in particles:
            for dx, dy in offsets:
                circ = Circle((p[0] + dx, p[1] + dy), radius=r,
                              edgecolor="white", facecolor="none",
                              lw=0.8, alpha=0.3, zorder=2)
                self.ax_left.add_patch(circ)
                self.fov_patches.append(circ)

    # =========================================================
    # Update loop
    # =========================================================
    def update(self, particles, potential,
               Fx_attr=None, Fy_attr=None,
               Fx_rep=None, Fy_rep=None,
               Fx_total=None, Fy_total=None,
               step=None):
        if not self.enabled or self.quit:
            return
        if self.paused:
            plt.pause(0.05)
            return

        # Left panel
        self.im.set_data(potential)
        self.im.set_clim(np.nanmin(potential), np.nanmax(potential))
        self.scat.set_offsets(particles)
        self._update_fov(particles)
        self.text.set_text(f"Step {step if step is not None else 0}")

        # Right panel
        if self.ax_right is not None:
            stride = self._quiv_stride
            if Fx_attr is not None:
                self.quiv_attr.set_UVC(Fx_attr[::stride, ::stride],
                                       Fy_attr[::stride, ::stride])
            if Fx_rep is not None:
                self.quiv_rep.set_UVC(Fx_rep[::stride, ::stride],
                                      Fy_rep[::stride, ::stride])
            if Fx_total is not None:
                self.quiv_tot.set_UVC(Fx_total[::stride, ::stride],
                                      Fy_total[::stride, ::stride])

            # overlay viewpoints
            self.scat_right.set_offsets(particles)
            # highlight selected viewpoint
            if len(particles) > 0:
                self.highlight.set_offsets([particles[self.i_selected]])
            else:
                self.highlight.set_offsets([])

        self._set_quiver_visibility()

        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
        plt.pause(self.target_dt)

    # =========================================================
    def close(self):
        if self.enabled:
            try:
                plt.ioff()
                plt.close(self.fig)
            except Exception:
                pass
