import numpy as np

from vispy import plot as vp

fig = vp.Fig(size=(600, 500), show=False)

# Plot the target square wave shape
x = np.linspace(0, 10, 1000)
colors = [(0.8, 0, 0, 1),
          (0.8, 0, 0.8, 1),
          (0, 0, 1.0, 1),
          (0, 0.7, 0, 1), ]

lines = []
for i,c in enumerate(colors):
    y = np.random.normal(0, 10, size=1000)
    tmp_line = fig[0, 0].plot((x, y+10*i), color=c, width=2)
    tmp_line.update_gl_state(depth_test=False)
    lines.append(tmp_line)
lines[0].set_data((x, y))
labelgrid = fig[0, 0].view.add_grid(margin=10)

grid = vp.visuals.GridLines(color=(0, 0, 0, 0.5))
grid.set_gl_state('translucent')
fig[0, 0].view.add(grid)


if __name__ == '__main__':
    fig.show(run=True)