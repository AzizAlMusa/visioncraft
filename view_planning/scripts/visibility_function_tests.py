import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# Define coverage_quality levels: (coverage_level, quality_value)
coverage_qualities = [
    (0.0, 0.0),
    (1.0, 1.0),
    (2.0, 1.0),
    (3.0, 1.0),
    (4.0, 1.0)
]
default_sharpness = 12.0

def compute_visibility_quality(C, qualities, sharpness):
    V = np.zeros_like(C)
    for i in range(len(qualities)):
        level, value = qualities[i]
        if i == len(qualities) - 1:
            weight = 1.0 / (1.0 + np.exp(-sharpness * (C - level)))
            V += value * weight
        else:
            next_level = qualities[i + 1][0]
            rising_edge = 1.0 / (1.0 + np.exp(-sharpness * (C - level)))
            falling_edge = 1.0 / (1.0 + np.exp(-sharpness * (C - next_level)))
            window = rising_edge * (1.0 - falling_edge)
            V += value * window
    return np.clip(V, 0.0, 1.0)

# X-axis range
C = np.linspace(0, 6, 1000)
initial_values = [q[1] for q in coverage_qualities]

# Plot setup
fig, ax = plt.subplots(figsize=(8, 4))
plt.subplots_adjust(left=0.1, bottom=0.4)
line, = ax.plot(C, compute_visibility_quality(C, coverage_qualities, default_sharpness), lw=2)
ax.set_xlabel('Effective Coverage C(x)')
ax.set_ylabel('Visibility Quality V(x)')
ax.set_ylim(-0.1, 1.1)
ax.set_title('Visibility Quality Function')

# Sliders for each quality level
sliders = []
for i, (level, val) in enumerate(coverage_qualities):
    ax_slider = plt.axes([0.1, 0.35 - i*0.05, 0.8, 0.03])
    slider = Slider(ax_slider, f'Q({level})', 0.0, 1.0, valinit=val)
    sliders.append(slider)

# Slider for sharpness
ax_sharpness = plt.axes([0.1, 0.05, 0.8, 0.03])
sharpness_slider = Slider(ax_sharpness, 'Sharpness (s)', 1.0, 30.0, valinit=default_sharpness)

def update(val):
    new_qualities = [(coverage_qualities[i][0], sliders[i].val) for i in range(len(sliders))]
    sharpness = sharpness_slider.val
    V = compute_visibility_quality(C, new_qualities, sharpness)
    line.set_ydata(V)
    fig.canvas.draw_idle()

for slider in sliders:
    slider.on_changed(update)
sharpness_slider.on_changed(update)

plt.show()
