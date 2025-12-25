import json
from matplotlib import pyplot as plt
from matplotlib import animation
import numpy as np

def normalize_t(y_old):
    x_old = np.linspace(0, 1, len(y_old))
    x_new = np.linspace(0, 1, 100)
    return np.interp(x_new, x_old, y_old)

with open('ten_strokes.json', 'r') as file:
    strokes = json.load(file)


poses_list = strokes[0]


poses_x = [[i[0] for i in poses_list[p]] for p in map(str,[12,14,16,18,24,26,28])]
for i in range(len(poses_x)):
    poses_x[i] = normalize_t(poses_x[i])
poses_y = [[i[1] for i in poses_list[p]] for p in map(str,[12,14,16,18,24,26,28])]
for i in range(len(poses_y)):
    poses_y[i] = normalize_t(poses_y[i])


def update(i):
    if i < len(poses_x[0]):
        graph.set_offsets([
    (poses_x[j][i], poses_y[j][i])
    for j in range(len(poses_x))
])

    else:
        pass

# create N_points initial points
fig = plt.figure(figsize=(5, 5))
ax = fig.add_subplot(111)
ax.set_xlim(0,1)
ax.set_ylim(0,1)
graph = ax.scatter([s[0] for s in poses_x],  [s[1] for s in poses_y], s=7, color='orange')

# Creating the Animation object
ani = animation.FuncAnimation(fig, update, frames=len(poses_x[0])-1, interval=25, blit=False)
plt.show()
