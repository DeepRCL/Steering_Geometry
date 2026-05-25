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

# Early steps: long + wide wobble; later steps: short + tight, nearly straight
path = np.array([
    [0.67, 0.67],   # start — red peak
    [0.74, 0.52],   # long step, big rightward wobble
    [0.50, 0.47],   # long step, big leftward wobble
    # [0.57, 0.38],   # medium step, mild right
    [0.42, 0.34],   # medium step, mild left
    [0.38, 0.28],   # shorter, less wobble
    [0.28, 0.26],   # short, nearly straight
    [0.22, 0.23],   # very short, straight
    # [0.18, 0.21],   # tiny final step
])

for i in range(len(path) - 1):
    p0 = path[i]
    p1 = path[i + 1]
    ax.annotate(
        '', xy=p1, xytext=p0,
        arrowprops=dict(
            arrowstyle='->', color='#000000',
            lw=4.5,
            mutation_scale=24,
            connectionstyle='arc3,rad=0.0'
        )
    )

for i, pt in enumerate(path):
    size = max(3.5, 8.5 - i * 0.5)
    ax.plot(pt[0], pt[1], 'o', color='#000000', markersize=size, zorder=5)

ax.text(
    0.97, 0.04,
    r'Guided By $-\nabla\mathcal{L}$',
    transform=ax.transAxes,
    fontsize=32,
    color='#C0392B',
    fontweight='bold',
    zorder=10,
    verticalalignment='bottom',
    horizontalalignment='right',
)

ax.set_aspect('equal')
ax.axis('off')
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
plt.tight_layout()
plt.savefig('contour6.png', dpi=300, bbox_inches='tight')
print("saved")
