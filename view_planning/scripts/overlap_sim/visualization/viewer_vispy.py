# visualization/viewer_vispy.py
from vispy import app, scene
import numpy as np
import cupy as cp

class FastViewer:
    """Fast VisPy viewer for potential + viewpoint overlay."""
    def __init__(self, grid_size=100):
        self.canvas = scene.SceneCanvas(keys='interactive',
                                        show=True,
                                        bgcolor='black',
                                        size=(800, 800))
        self.view = self.canvas.central_widget.add_view()
        self.view.camera = scene.cameras.PanZoomCamera(aspect=1)
        self.view.camera.set_range(x=(0, grid_size), y=(0, grid_size))

        self.image = scene.visuals.Image(np.zeros((grid_size, grid_size),
                                                  dtype=np.float32),
                                         cmap='plasma',
                                         parent=self.view.scene)
        self.scatter = scene.visuals.Markers(parent=self.view.scene)
        self.scatter.set_data(np.empty((0, 2)),
                      face_color='white', size=6)

        # Maintain compatibility with the modular viewer interface
        self.active_index = 0
        self.insertion_request = False


    def update(self, data):
        pot = data.get("potential")
        pts = data.get("particles")

        # Handle CuPy or NumPy arrays
        if isinstance(pot, cp.ndarray):
            pot = cp.asnumpy(pot)
        if isinstance(pts, cp.ndarray):
            pts = cp.asnumpy(pts)

        self.image.set_data(pot)
        if pts.size:
            self.scatter.set_data(pts, face_color='white', size=6)

        self.canvas.update()
        app.process_events()

    def close(self):
        self.canvas.close()
