# visualization/panels/potential_panel.py
from vispy import scene
import numpy as np


class PotentialPanel:
    """GPU potential field panel with toroidal FOV circles and visible viewpoint dots."""

    def __init__(self, grid_size=100, fov_radius=20.0):
        self.grid_size = grid_size
        self.fov_radius = fov_radius
        self._first_real_draw = False

        # --- Main view --------------------------------------------------------
        self.view = scene.widgets.ViewBox(border_color='gray')
        self.view.bgcolor = 'black'
        self.view.camera = scene.PanZoomCamera(aspect=1)
        self.view.camera.set_range(x=(0, grid_size), y=(0, grid_size), margin=0.0)
        self.view.camera.flip = (0, 1, 0)

        # --- Potential image (VisPy texture expects NumPy) --------------------
        init_img = np.zeros((grid_size, grid_size), dtype=np.float32)
        self.image = scene.visuals.Image(
            init_img,
            cmap='plasma',
            interpolation='nearest',
            clim=(0.0, 1.0),
            parent=self.view.scene,
        )

        # --- Viewpoint markers ------------------------------------------------
        dummy = np.zeros((1, 2), dtype=np.float32)   # shape=(N,2) required
        self.scatter = scene.visuals.Markers(parent=self.view.scene)
        self.scatter.set_data(
            dummy,
            face_color='white',
            edge_color='black',
            size=8,
            symbol='disc'
        )
        self.scatter.set_gl_state(
            depth_test=False,
            blend=True,
            blend_func=('src_alpha', 'one_minus_src_alpha')
        )
        self.scatter.transform = scene.transforms.STTransform(translate=(0, 0, -1e-3))

        # --- Active-viewpoint highlight --------------------------------------
        self.highlight = scene.visuals.Markers(parent=self.view.scene)
        self.highlight.set_data(
            dummy,
            face_color=(1.0, 0.3, 0.1, 1.0),
            size=12,
            symbol='star'
        )
        self.highlight.set_gl_state(
            depth_test=False,
            blend=True,
            blend_func=('src_alpha', 'one_minus_src_alpha')
        )
        self.highlight.transform = scene.transforms.STTransform(translate=(0, 0, -1e-3))

        # --- Batched FOV circle lines ----------------------------------------
        self.circle_lines = scene.visuals.Line(
            pos=np.array([[0.0, 0.0]], dtype=np.float32),
            color=(1, 1, 1, 0.45),
            width=1.2,
            method='gl',
            parent=self.view.scene,
        )
        self.circle_lines.order = 1
        self.circle_lines.set_gl_state(
            depth_test=False,
            blend=True,
            blend_func=('src_alpha', 'one_minus_src_alpha')
        )
        self.circle_lines.transform = scene.transforms.STTransform(translate=(0, 0, -1e-3))

        # --- Optional grid and title -----------------------------------------
        self.grid = scene.visuals.GridLines(
            color=(0.3, 0.3, 0.3, 0.4),
            parent=self.view.scene
        )
        self.title = scene.visuals.Text(
            'Potential Field',
            color='white',
            font_size=12,
            anchor_x='center',
            anchor_y='top',
            pos=(grid_size / 2, -10),
            parent=self.view.scene,
        )

    # -------------------------------------------------------------------------
    def _update_fov_lines(self, pts):
        """Update batched circle line vertices for all viewpoints (toroidal wrap)."""
        if pts is None or len(pts) == 0:
            self.circle_lines.set_data(pos=np.array([[0.0, 0.0]], dtype=np.float32))
            return

        r = float(self.fov_radius)
        g = float(self.grid_size)
        n_seg = 64
        theta = np.linspace(0.0, 2.0 * np.pi, n_seg, endpoint=True, dtype=np.float32)
        unit_circle = np.stack((np.cos(theta), np.sin(theta)), axis=1)  # (n_seg, 2)

        all_vertices = []
        for (x, y) in pts:
            # 9 offset positions for toroidal wrapping
            for dx in (-g, 0.0, g):
                for dy in (-g, 0.0, g):
                    cx, cy = x + dx, y + dy
                    # skip circles completely outside visible range
                    if (cx + r < 0) or (cy + r < 0) or (cx - r > g) or (cy - r > g):
                        continue
                    ring = unit_circle * r + np.array([cx, cy], dtype=np.float32)
                    ring = np.vstack([ring, ring[0]])  # close loop
                    all_vertices.append(ring)

        if not all_vertices:
            self.circle_lines.set_data(pos=np.array([[0.0, 0.0]], dtype=np.float32))
            return

        verts = np.vstack(all_vertices).astype(np.float32, copy=False)

        # connection indices for independent loops
        connect = []
        offset = 0
        for ring in all_vertices:
            N = len(ring)
            segs = np.column_stack(
                [np.arange(offset, offset + N - 1, dtype=np.uint32),
                 np.arange(offset + 1, offset + N, dtype=np.uint32)]
            )
            connect.append(segs)
            offset += N
        connect = np.vstack(connect).astype(np.uint32, copy=False)

        self.circle_lines.set_data(
            pos=verts,
            connect=connect,
            color=(1, 1, 1, 0.45),
            width=1.2,
        )

    # -------------------------------------------------------------------------
    def update(self, data: dict):
        """Update potential image and overlay viewpoints (robust CuPy/NumPy support)."""
        # Import CuPy locally and convert explicitly only here
        try:
            import cupy as cp
        except Exception:
            cp = None

        pot = data.get("potential", None)
        pts = data.get("particles", None)

        if pot is None:
            return

        # --- Explicit GPU→CPU copy (single point of conversion) --------------
        if cp is not None and isinstance(pot, cp.ndarray):
            pot = cp.asnumpy(pot)
        pot = np.asarray(pot, dtype=np.float32, order="C")
        np.nan_to_num(pot, copy=False)

        # --- Normalize to [0,1] for guaranteed visible contrast --------------
        pmin = float(pot.min())
        pmax = float(pot.max())
        rng = pmax - pmin
        if not np.isfinite(rng) or rng < 1e-12:
            pot_norm = np.zeros_like(pot, dtype=np.float32)
        else:
            pot_norm = (pot - pmin) / rng

        # --- Update image texture and color limits ---------------------------
        self.image.set_data(pot_norm)
        self.image.clim = (0.0, 1.0)

        # --- Viewpoint markers and FOV circles -------------------------------
        if pts is not None and len(pts):
            if cp is not None and isinstance(pts, cp.ndarray):
                pts = cp.asnumpy(pts)
            pts = np.asarray(pts, dtype=np.float32, order="C")

            self.scatter.set_data(
                pts,
                face_color='white',
                edge_color='black',
                size=8,
                symbol='disc'
            )
            self.highlight.set_data(
                pts[[0]],
                face_color=(1.0, 0.3, 0.1, 1.0),
                size=12,
                symbol='star'
            )
            self._update_fov_lines(pts)

        # --- One-time camera range ------------------------------------------
        if not self._first_real_draw:
            H, W = pot.shape
            self.view.camera.set_range(x=(0, W), y=(0, H), margin=0.0)
            self._first_real_draw = True

        # --- Safe redraw -----------------------------------------------------
        canvas = getattr(self.view, "canvas", None)
        if canvas and not getattr(canvas, "_closed", False):
            try:
                canvas.update()
                canvas.render()
            except Exception:
                setattr(canvas, "_closed", True)
