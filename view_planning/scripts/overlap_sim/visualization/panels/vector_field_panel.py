# visualization/panels/vector_field_panel.py
from vispy import scene
import numpy as np
import cupy as cp


class VectorFieldPanel:
    """Right panel: GPU vector fields (attr / rep / total) with magnitude-scaled
    oriented triangle arrowheads, capped magnitudes, and viewpoint overlays."""

    def __init__(self, grid_size=100, stride=None):
        self.grid_size = int(grid_size)
        if stride is None:
            stride = max(1, self.grid_size // 20)
        self.stride = int(stride)

        # === Rendering configuration ========================================
        self.config = {
            "max_force_mag": 20.0,   # maximum vector length
            "vec_scale": 1.0,        # visual scale multiplier
            "head_length": 2.0,      # base arrowhead length
            "head_width": 1.75,      # base arrowhead width
            "line_width": 1.6,       # stem thickness
            "fade_old": 0.7,         # motion smoothing
        }

        # === Sampling grid ===================================================
        xs = np.arange(0, self.grid_size, self.stride, dtype=np.float32)
        ys = np.arange(0, self.grid_size, self.stride, dtype=np.float32)
        Xg, Yg = np.meshgrid(xs, ys)
        self._starts = np.column_stack([Xg.ravel(), Yg.ravel()])

        # === View setup ======================================================
        self.view = scene.widgets.ViewBox(border_color='gray')
        self.view.bgcolor = 'black'
        self.view.camera = scene.PanZoomCamera(aspect=1)
        self.view.camera.set_range(x=(0, grid_size), y=(0, grid_size), margin=0.0)
        self.view.camera.flip = (0, 1, 0)

        # === Line fields (stems) ============================================
        self.fields = {
            "attr":  scene.visuals.Line(color=(0.0, 0.7, 1.0, 0.8),
                                        width=self.config["line_width"], method='gl', parent=self.view.scene),
            "rep":   scene.visuals.Line(color=(1.0, 0.2, 0.7, 0.8),
                                        width=self.config["line_width"], method='gl', parent=self.view.scene),
            "total": scene.visuals.Line(color=(0.0, 1.0, 0.7, 0.85),
                                        width=self.config["line_width"], method='gl', parent=self.view.scene),
        }
        for vis in self.fields.values():
            vis.order = 1
            vis.set_gl_state(depth_test=False, blend=True,
                             blend_func=('src_alpha', 'one_minus_src_alpha'))

        # === Arrowhead meshes ===============================================
        self.arrow_heads = {
            k: scene.visuals.Mesh(color=v.color, parent=self.view.scene)
            for k, v in self.fields.items()
        }
        for vis in self.arrow_heads.values():
            vis.order = 2
            vis.set_gl_state(depth_test=False, blend=True,
                             blend_func=('src_alpha', 'one_minus_src_alpha'))

        # === Viewpoint markers ===============================================
        self.viewpoints = scene.visuals.Markers(parent=self.view.scene)
        self.viewpoints.set_data(np.array([[0.0, 0.0]]),
                                 face_color='white', edge_color='black',
                                 size=8, symbol='disc')
        self.viewpoints.order = 3

        self.highlight = scene.visuals.Markers(parent=self.view.scene)
        self.highlight.set_data(np.array([[0.0, 0.0]]),
                                face_color=(0, 0, 0, 0),
                                edge_color='yellow', size=14, symbol='o')
        self.highlight.order = 4
        self.highlight.visible = False

        # === Force arrows for active viewpoint ===============================
        self.force_arrows = {
            "attr": scene.visuals.Line(color=(0.0, 0.9, 1.0, 0.8),
                                       width=2.0, method='gl', parent=self.view.scene),
            "rep":  scene.visuals.Line(color=(1.0, 0.3, 0.7, 0.8),
                                       width=2.0, method='gl', parent=self.view.scene),
            "total": scene.visuals.Line(color=(0.0, 1.0, 0.7, 0.9),
                                        width=2.0, method='gl', parent=self.view.scene),
        }
        for a in self.force_arrows.values():
            a.order = 5
            a.visible = True

        # === Title ===========================================================
        self.title = scene.visuals.Text(
            'Vector Fields (A=Attr, R=Rep, T=Total, B=Both, N=Next VP)',
            color='white', font_size=12,
            anchor_x='center', anchor_y='top',
            pos=(grid_size / 2, -10),
            parent=self.view.scene,
        )

        # === State tracking ==================================================
        self.mode = "total"
        self.i_selected = 0
        self._last_V = {
            "attr": np.zeros_like(self._starts, dtype=np.float32),
            "rep":  np.zeros_like(self._starts, dtype=np.float32),
            "total": np.zeros_like(self._starts, dtype=np.float32),
        }

    # -------------------------------------------------------------------------
    # -------------------------------------------------------------------------
    def _get_field(self, data, key_prefix):
        """Fetch Fx,Fy arrays and move from GPU→CPU only if needed."""
        import cupy as cp

        Fx = data.get(f"Fx_{key_prefix}")
        Fy = data.get(f"Fy_{key_prefix}")
        if Fx is None or Fy is None:
            return None, None

        # ---- single CuPy→NumPy copy for this field --------------------------
        if isinstance(Fx, cp.ndarray):
            Fx = cp.asnumpy(Fx)
        if isinstance(Fy, cp.ndarray):
            Fy = cp.asnumpy(Fy)

        Fx_s = Fx[::self.stride, ::self.stride]
        Fy_s = Fy[::self.stride, ::self.stride]

        # optional visibility weighting for attraction
        if key_prefix == "attr":
            need_map = data.get("need_map")
            if need_map is not None:
                if isinstance(need_map, cp.ndarray):
                    need_map = cp.asnumpy(need_map)
                need_s = need_map[::self.stride, ::self.stride]
                Fx_s *= need_s
                Fy_s *= need_s
        return Fx_s.astype(np.float32), Fy_s.astype(np.float32)


    # -------------------------------------------------------------------------
    def _build_vectors(self, Fx, Fy, last_V):
        """Smooth, scale, and cap vector lengths."""
        V = np.column_stack([Fx.ravel(), Fy.ravel()]).astype(np.float32)
        V *= self.config["vec_scale"]

        norms = np.linalg.norm(V, axis=1, keepdims=True)
        mask = norms[:, 0] > 1e-12
        if mask.any():
            capped = np.minimum(norms[mask], self.config["max_force_mag"])
            V[mask] = V[mask] / norms[mask] * capped
        else:
            V[:] = 0.0

        fade = self.config["fade_old"]
        V = fade * last_V + (1.0 - fade) * V
        last_V[:] = V

        start = self._starts
        end = start + V
        verts = np.vstack([start, end])
        n = len(start)
        connect = np.column_stack([np.arange(n), np.arange(n, 2 * n)])
        return verts, connect, start, end, V

    # -------------------------------------------------------------------------
    def _build_triangle_heads(self, end, V):
        """Construct oriented triangle arrowheads scaled with vector magnitude."""
        if len(end) == 0:
            return np.zeros((0, 3), np.float32), np.zeros((0, 3), np.uint32)

        # Direction and magnitude
        norms = np.linalg.norm(V, axis=1, keepdims=True) + 1e-12
        dirs = V / norms

        # Scale heads proportionally to magnitude
        mag_ratio = np.clip(norms / self.config["max_force_mag"], 0.15, 1.0)
        head_len = self.config["head_length"] * mag_ratio
        head_wid = self.config["head_width"] * mag_ratio

        perp = np.column_stack([-dirs[:, 1], dirs[:, 0]])

        v0 = end
        v1 = end - dirs * head_len + perp * head_wid * 0.5
        v2 = end - dirs * head_len - perp * head_wid * 0.5

        vertices = np.vstack([v0, v1, v2]).astype(np.float32)
        n = len(end)
        faces = np.column_stack([
            np.arange(0, n),
            np.arange(n, 2 * n),
            np.arange(2 * n, 3 * n)
        ]).astype(np.uint32)

        return vertices, faces

    # -------------------------------------------------------------------------
    def _set_visibility(self):
        self.fields["attr"].visible = self.mode in ["attr", "both"]
        self.fields["rep"].visible = self.mode in ["rep", "both"]
        self.fields["total"].visible = self.mode == "total"
        for k in ["attr", "rep", "total"]:
            self.arrow_heads[k].visible = self.fields[k].visible
            self.force_arrows[k].visible = self.fields[k].visible

    # -------------------------------------------------------------------------
    def toggle(self, key):
        key = key.lower()
        modes = {"a": "attr", "r": "rep", "t": "total", "b": "both"}
        if key in modes:
            self.mode = modes[key]
            print(f"[VectorFieldPanel] Mode={self.mode}")
            self._set_visibility()
        elif key == "n":
            self.i_selected += 1
            print(f"[VectorFieldPanel] Cycle viewpoint highlight -> {self.i_selected}")

    # -------------------------------------------------------------------------
    def _cap_force(self, vec):
        mag = np.linalg.norm(vec)
        if mag > self.config["max_force_mag"]:
            return vec * (self.config["max_force_mag"] / (mag + 1e-12))
        return vec

    # -------------------------------------------------------------------------
    def update(self, data: dict):
        stride = self.stride
        particles = data.get("particles", [])
        i_active = data.get("active_index", 0)
        k_rep = data.get("k_rep", 1.0)

        self._last_particle_count = len(particles)
        if len(particles) > 0:
            i_active = i_active % len(particles)

        # === Vector fields and oriented heads ===
        for name in ["attr", "rep", "total"]:
            Fx, Fy = self._get_field(data, name)
            if Fx is None or Fy is None:
                continue
            verts, connect, start, end, V = self._build_vectors(Fx, Fy, self._last_V[name])
            self.fields[name].set_data(pos=verts, connect=connect,
                                       width=self.config["line_width"])
            vertices, faces = self._build_triangle_heads(end, V)
            self.arrow_heads[name].set_data(vertices=vertices, faces=faces,
                                            color=self.fields[name].color)

        # === Viewpoint markers ===
        if len(particles) > 0:
            if isinstance(particles, cp.ndarray):
                particles = cp.asnumpy(particles)
            self.viewpoints.set_data(particles, face_color='white',
                                     edge_color='black', size=8, symbol='disc')
            self.i_selected %= len(particles)
            self.highlight.set_data(particles[self.i_selected:self.i_selected + 1])
            self.highlight.visible = True
        else:
            self.highlight.visible = False

        # === Per-viewpoint resultant arrows ===
        F_attr = data.get("F_attr")
        F_rep = data.get("F_rep")
        F_total = data.get("F_total")
        if F_attr is not None and len(particles) > 0:
            p = particles[i_active]
            px, py = p
            fx, fy = self._cap_force(F_attr[i_active])
            self.force_arrows["attr"].set_data(
                pos=np.array([[px, py], [px + fx * 2.0, py + fy * 2.0]], np.float32))
            fx, fy = self._cap_force(k_rep * F_rep[i_active])
            self.force_arrows["rep"].set_data(
                pos=np.array([[px, py], [px + fx * 2.0, py + fy * 2.0]], np.float32))
            fx, fy = self._cap_force(F_total[i_active])
            self.force_arrows["total"].set_data(
                pos=np.array([[px, py], [px + fx * 2.0, py + fy * 2.0]], np.float32))

        self._set_visibility()

        # redraw (skip if canvas already destroyed or size invalid)
        canvas = getattr(self.view, "canvas", None)
        if canvas and not getattr(canvas, "_closed", False):
            try:
                canvas.update(); canvas.render()
            except Exception:
                setattr(canvas, "_closed", True)



