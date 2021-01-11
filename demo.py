import tikzplotlib
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons
from matplotlib.collections import LineCollection
import matplotlib.animation as animation
from tqdm import tqdm

import generate
import torch
import numpy as np
import os

np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
# live is 0
live_or_animate= True

def generate_rotated_hexagon(theta, hexagon_segments):
    rotation_matrix = generate.generate_rotation(theta)
    # [segments, points, [x,y]]
    new_segments = torch.einsum("ab,cdb->cda", rotation_matrix, hexagon_segments)
    return new_segments

max_radius = 30

fig = plt.figure()
if not live_or_animate:
    plt.subplots_adjust(bottom=plt.rcParams["figure.subplot.bottom"] + 0.1)
ax = fig.add_subplot(1, 1, 1)
# ax = plt.gca()
ax.set_aspect("equal")
max_dist = max_radius*1.1+1
ax.set_xlim(-max_dist, max_dist)
ax.set_ylim(-max_dist, max_dist)

# [num_segments, 2 points, 2 coords]
hexagon_segments = generate.generate_hexagon(max_radius, theta=0)
hexagon_segments[...,1]=hexagon_segments[...,1]*1.01

collection_a = LineCollection(
    generate_rotated_hexagon(
        torch.as_tensor(0, dtype=torch.double, device=generate.device),
        hexagon_segments,
    ).cpu(),
    colors=["black"],
    zorder=0,
)
collection_b = LineCollection(
    generate_rotated_hexagon(
        torch.as_tensor(0, dtype=torch.double, device=generate.device),
        hexagon_segments,
    ).cpu(),
    colors=["black"],
    zorder=0,
)
text = ax.text(-max_dist, max_dist, np.float(0))

ax.add_collection(collection_a)
ax.add_collection(collection_b)


def update(val):
    angle = torch.as_tensor(np.deg2rad(val), dtype=torch.double, device=generate.device)
    collection_a.set_segments(generate_rotated_hexagon(angle/2, hexagon_segments).cpu())
    collection_b.set_segments(generate_rotated_hexagon(-angle/2, hexagon_segments).cpu())
    text.set_text(val)

    if not live_or_animate:
        fig.canvas.draw_idle()
    else:
        return [collection_a, collection_b, text]

if not live_or_animate:
    axcolor = "lightgoldenrodyellow"
    angle_ax = plt.axes(
        [
            plt.rcParams["figure.subplot.left"],
            plt.rcParams["figure.subplot.bottom"],
            plt.rcParams["figure.subplot.right"] - plt.rcParams["figure.subplot.left"],
            0.03,
        ],
        facecolor=axcolor,
    )
    angle_slider = Slider(angle_ax, "Angle", 0, 360, valinit=0, valstep=1)

    angle_slider.on_changed(update)
else:
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)
    ani = animation.FuncAnimation(fig, update, tqdm(np.linspace(0,360,num=360*10)), interval=1, blit=True)
ani.save('MoireLapseDilated1percent.mp4', writer=writer)

# plt.show()
