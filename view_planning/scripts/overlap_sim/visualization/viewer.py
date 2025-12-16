# visualization/viewer.py
from vispy import app, scene
from vispy.scene import widgets
import imageio_ffmpeg
import numpy as np
import os
import subprocess, tempfile


class Viewer:
    """Modular multi-panel GPU viewer with optional video recording.

    Features:
        • VisPy-based multi-panel layout (supports any number of panels)
        • Safe keyboard handling and camera initialization
        • Graceful window-closing logic (no render after destruction)
        • Reliable MP4 export using direct ffmpeg writer
    """

    def __init__(self, panels, title="OverlapSim Viewer", size=(1200, 600),
                 record=False, out_path="./videos/output.mp4", fps=30):
        # === Canvas setup =====================================================
        self.canvas = scene.SceneCanvas(
            keys='interactive',
            show=True,
            title=title,
            bgcolor='black',
            size=size,
        )

        # === Layout grid for panels ===========================================
        self.grid = widgets.Grid()
        self.canvas.central_widget.add_widget(self.grid)
        self.panels = panels
        self.canvas.events.key_press.connect(self.on_key_press)
        self.canvas.events.close.connect(self._on_close)

        for i, p in enumerate(self.panels):
            self.grid.add_widget(p.view, row=0, col=i)

        # === State tracking ====================================================
        self.active_index = 0
        self.insertion_request = False
        self._closed = False

        # === Recording configuration ==========================================
        self.record = record
        self.out_path = out_path
        self.fps = fps
        self._frames = [] if record else None

        # === Camera initialization ============================================
        for p in self.panels:
            if hasattr(p, "grid_size") and hasattr(p, "view") and hasattr(p.view, "camera"):
                cam = p.view.camera
                cam.set_range(x=(0, p.grid_size), y=(0, p.grid_size), margin=0.0)
                cam.zoom(1.0)
                if hasattr(cam, "flip"):
                    cam.flip = (0, 1, 0)

        # First draw to ensure OpenGL context initialized
        self.canvas.update()
        self.canvas.render()
        app.process_events()

        # Prepare video directory
        if self.record:
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            print(f"[Viewer] Recording enabled → {out_path}")

    # --------------------------------------------------------------------------
    def _on_close(self, event=None):
        """Triggered by VisPy close event."""
        self._closed = True

    # --------------------------------------------------------------------------
    def on_key_press(self, event):
        """Handle keyboard shortcuts for all panels and viewer-level actions."""
        key = event.key.name.lower()
        for panel in self.panels:
            if hasattr(panel, "toggle"):
                panel.toggle(key)

        if key == "n":
            for p in self.panels:
                if hasattr(p, "i_selected"):
                    if hasattr(p, "_last_particle_count") and p._last_particle_count > 0:
                        p.i_selected = p.i_selected % p._last_particle_count
                    self.active_index = p.i_selected
                    break
        elif key == "i":
            self.insertion_request = True
            print("[Viewer] Manual insertion requested")
        elif key == "m":
            self.active_index += 1
            print(f"[Viewer] Active viewpoint manually set to {self.active_index}")
        elif key == "escape":
            self.close()

    # --------------------------------------------------------------------------
    def _safe_render(self):
        """Render only if canvas and context are still valid."""
        if self._closed or getattr(self.canvas, "native", None) is None:
            return False
        try:
            self.canvas.update()
            self.canvas.render()
            app.process_events()
            return True
        except Exception:
            self._closed = True
            return False

    # --------------------------------------------------------------------------
    def _capture_frame(self):
        """Safely capture current frame for recording."""
        if not self.record or self._closed:
            return
        try:
            if getattr(self.canvas, "native", None) is None:
                return
            size = getattr(self.canvas, "size", None)
            if size is None or size == (0, 0):
                return
            frame = self.canvas.render()
            if frame is not None:
                frame = np.asarray(frame)[..., :3]  # remove alpha
                self._frames.append(frame)
        except Exception:
            pass

    # --------------------------------------------------------------------------
    def update(self, data):
        """Update all panels and optionally record the current frame."""
        if self._closed:
            return
        for panel in self.panels:
            try:
                panel.update(data)
            except Exception:
                continue
        if not self._safe_render():
            return
        self._capture_frame()

    # --------------------------------------------------------------------------
    def _save_video_mp4(self):
        """Encode stored frames via imageio_ffmpeg with correct aspect and high quality."""
        if not self._frames:
            print("[Viewer] No frames captured, skipping video save.")
            return

        print(f"[Viewer] Saving video with {len(self._frames)} frames → {self.out_path}")
        try:
            import cv2
        except ImportError:
            raise RuntimeError("OpenCV (cv2) required; install with `pip install opencv-python`")

        try:
            height, width, _ = self._frames[0].shape
            # Compute 1080p width preserving aspect ratio
            target_h = 1080
            target_w = int(width * (target_h / height))
            if target_w % 2 != 0:  # enforce even width for H.264
                target_w += 1

            process = imageio_ffmpeg.write_frames(
                self.out_path,
                size=(target_w, target_h),
                fps=60,
                codec="libx264",
                quality=8,          # let ffmpeg pick CRF≈23 equivalent
                output_params=[
                    "-crf", "18",   # visually lossless
                    "-preset", "slow",
                    "-pix_fmt", "yuv420p"
                ],
            )
            process.send(None)

            for frame in self._frames:
                frame = np.ascontiguousarray(frame, dtype=np.uint8)
                # upscale only if smaller, preserving aspect
                frame_up = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)
                process.send(frame_up)

            process.close()
            print(f"[Viewer] Video saved successfully ({target_w}×{target_h} @ 60 fps) → {self.out_path}")
        except Exception as e:
            print(f"[Viewer] MP4 save failed: {e}")



    # --------------------------------------------------------------------------
    def close(self):
        """Graceful shutdown + MP4 export."""
        if self._closed:
            return
        self._closed = True
        if self.record and self._frames:
            self._save_video_mp4()
        try:
            self.canvas.close()
        except Exception:
            pass
