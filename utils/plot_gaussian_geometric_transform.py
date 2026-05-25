import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.collections import LineCollection

N = 500
x = np.linspace(0, 1, N)
y = np.linspace(0, 1, N)
X, Y = np.meshgrid(x, y)

def gaussian2d(X, Y, mx, my, sx, sy, rho):
    dx = (X - mx) / sx
    dy = (Y - my) / sy
    z = dx**2 - 2*rho*dx*dy + dy**2
    return np.exp(-z / (2*(1 - rho**2)))

Z = (0.45 * gaussian2d(X, Y, 0.32, 0.33, 0.13, 0.11, -0.3)
   + 0.55 * gaussian2d(X, Y, 0.67, 0.67, 0.16, 0.12,  0.35))

Z1 = gaussian2d(X, Y, 0.32, 0.33, 0.13, 0.11, -0.3)
Z2 = gaussian2d(X, Y, 0.67, 0.67, 0.16, 0.12,  0.35)
blend = (0.55 * Z2) / (0.45 * Z1 + 0.55 * Z2 + 1e-10)

blend_cmap = LinearSegmentedColormap.from_list(
    'green_red', ['#27ae60', '#f39c12', '#e74c3c'], N=512
)

fig, ax = plt.subplots(figsize=(7, 7))
levels = np.linspace(Z.min() + 0.04*(Z.max()-Z.min()), Z.max() * 0.92, 14)

cs = ax.contour(X, Y, Z, levels=levels)
ax.cla()

for i, level_segs in enumerate(cs.allsegs):
    t = i / (len(levels) - 1)
    alpha = 0.55 + 0.45 * t
    lw = 0.9 + 0.5 * t
    for seg in level_segs:
        if len(seg) < 2:
            continue
        xi = np.clip((seg[:, 0] * (N-1)).astype(int), 0, N-1)
        yi = np.clip((seg[:, 1] * (N-1)).astype(int), 0, N-1)
        w_vals = blend[yi, xi]
        points = seg.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        seg_w = (w_vals[:-1] + w_vals[1:]) / 2
        colors = blend_cmap(seg_w)
        colors[:, 3] = alpha
        lc = LineCollection(segments, colors=colors, linewidth=lw,
                            capstyle='round', joinstyle='round')
        ax.add_collection(lc)

# Red peak center
target = [0.32, 0.33]

# Four paths converging to red center from different sections around it
# (start_point, rad) — positive rad curves counter-clockwise, negative clockwise
paths = [
    ([0.65, 0.78], [0.32, 0.33],  0.35),   # top-left, sweeps right
    # ([0.70, 0.77], [0.33, 0.31], -0.35),   # top-right, sweeps left
    ([0.75, 0.62], [0.34, 0.33], -0.25),   # right, curves upward
    ([0.68, 0.69], [0.33, 0.34],  0.30),   # below, curves right
]

for p0, p1, rad in paths:
    ax.annotate('', xy=p1, xytext=p0,
                arrowprops=dict(
                    arrowstyle='->',
                    color='#000000',
                    lw=3.2,
                    mutation_scale=26,
                    connectionstyle=f'arc3,rad={rad}',
                ))
    # Start dot
    ax.plot(p0[0], p0[1], 'o', color='#000000', markersize=7, zorder=5)

# Center dot at red peak
ax.plot(target[0], target[1], 'o', color='#000000', markersize=9, zorder=6)

ax.text(
    1, 0.04,
    'Guided By Geometric \n Transformations',
    transform=ax.transAxes,
    fontsize=26,
    color='#1E8449',
    fontweight='bold',
    zorder=10,
    multialignment='center',
    verticalalignment='bottom',
    horizontalalignment='right',
    linespacing=1.3,
)

ax.set_aspect('equal')
ax.axis('off')
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
plt.tight_layout()
plt.savefig('guided_by_distribution.png', dpi=300, bbox_inches='tight')
print("saved")