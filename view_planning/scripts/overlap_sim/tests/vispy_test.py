# test_vispy.py
from vispy import app, scene
print("VisPy imported successfully!")

canvas = scene.SceneCanvas(keys='interactive', show=True, bgcolor='black')
view = canvas.central_widget.add_view()
canvas.show()
app.run()
