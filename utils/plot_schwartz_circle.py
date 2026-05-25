import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from matplotlib.patches import Circle, Wedge


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from CAA.Geometry.config import GROUP_COLORS, HIGHER_ORDER_GROUPS, SCHWARTZ_CIRCUMPLEX_ORDER, value_to_group


BOUNDARY_GROUPS = {
    "Hedonism": ("Openness to Change", "Self-Enhancement"),
    "Face": ("Self-Enhancement", "Conservation"),
    "Humility": ("Conservation", "Self-Transcendence"),
}

STEER_TARGET = "Self-direction: action"
RING_R = 1.0
NODE_R = 0.06
LABEL_PAD = 0.13


def theoretical_circle_points(num_values: int) -> np.ndarray:
    angles = np.linspace(0.0, 2.0 * np.pi, num_values, endpoint=False)
    return np.column_stack([np.cos(angles), np.sin(angles)])


def short_value_label(value: str) -> str:
    return value.split(":")[-1].strip()


def draw_value_marker(ax, x: float, y: float, value: str, radius: float = NODE_R, highlighted: bool = False) -> None:
    zorder = 5 if highlighted else 4
    if value in BOUNDARY_GROUPS:
        left_group, right_group = BOUNDARY_GROUPS[value]
        ax.add_patch(Wedge((x, y), radius, 90, 270, facecolor=GROUP_COLORS[left_group], edgecolor="none", zorder=zorder))
        ax.add_patch(Wedge((x, y), radius, -90, 90, facecolor=GROUP_COLORS[right_group], edgecolor="none", zorder=zorder))
        ax.add_patch(Circle((x, y), radius, facecolor="none", edgecolor="white", linewidth=1.2, zorder=zorder + 1))
    else:
        ax.add_patch(
            Circle(
                (x, y),
                radius,
                facecolor=GROUP_COLORS.get(value_to_group(value), "black"),
                edgecolor="white",
                linewidth=1.2,
                zorder=zorder,
            )
        )

    if highlighted:
        ax.add_patch(Circle((x, y), radius * 1.55, facecolor="none", edgecolor="#D85A30", linewidth=2.3, zorder=zorder + 2))

fig, ax = plt.subplots(figsize=(7, 7))
ax.set_aspect("equal")
ax.axis("off")
ax.set_xlim(-1.55, 1.55)
ax.set_ylim(-1.55, 1.55)

# ── Dashed ring ────────────────────────────────────────────────────────────────
ring = plt.Circle((0, 0), RING_R, color="#aaa", fill=False,
                  linestyle="--", linewidth=1.0, zorder=1)
ax.add_patch(ring)

# ── Center dot ────────────────────────────────────────────────────────────────
ax.add_patch(Circle((0, 0), 0.04, color="#cccccc", zorder=3))

# ── Spokes, nodes, and labels ─────────────────────────────────────────────────
coords = theoretical_circle_points(len(SCHWARTZ_CIRCUMPLEX_ORDER))
positions = dict(zip(SCHWARTZ_CIRCUMPLEX_ORDER, coords))

for name, (xn, yn) in positions.items():
    highlighted = name == STEER_TARGET

    # spoke
    ax.plot([0, xn], [0, yn], color="#cccccc", linewidth=0.7,
            linestyle="--", zorder=1)

    # node circle
    draw_value_marker(ax, xn, yn, name, highlighted=highlighted)

    # label
    norm = np.hypot(xn, yn)
    xl = (RING_R + NODE_R + LABEL_PAD) * xn / norm
    yl = (RING_R + NODE_R + LABEL_PAD) * yn / norm
    ha = "center"
    if xl < -0.15:
        ha = "right"
    elif xl > 0.15:
        ha = "left"

    weight = "bold" if highlighted else "normal"
    fcolor = "#D85A30" if highlighted else GROUP_COLORS.get(value_to_group(name), "#333333")
    ax.text(xl, yl, short_value_label(name), ha=ha, va="center", fontsize=8.5,
            fontweight=weight, color=fcolor, zorder=5)

# ── Steering arrow (center → canonical Self-direction: action node) ───────────
xsd, ysd = positions[STEER_TARGET]
target_rad = np.arctan2(ysd, xsd)

# shorten the arrow so it ends just before the node circle
arrow_end_x = xsd - NODE_R * 1.8 * np.cos(target_rad)
arrow_end_y = ysd - NODE_R * 1.8 * np.sin(target_rad)

ax.annotate(
    "", xy=(arrow_end_x, arrow_end_y), xytext=(0.04, 0.0),
    arrowprops=dict(
        arrowstyle="-|>",
        color="#D85A30",
        lw=2.5,
        mutation_scale=18,
    ),
    zorder=6,
)

# ── Legend ────────────────────────────────────────────────────────────────────
legend_patch = mpatches.FancyArrow(
    0, 0, 0.1, 0, width=0.005,
    color="#D85A30", length_includes_head=True, head_width=0.02
)
ax.legend(
    handles=[
        mpatches.Patch(color="none", label=""),          # spacer
        plt.Line2D([0], [0], color="#D85A30", linewidth=2.5,
                   marker=">", markersize=8,
                   label="Steering vector (Δv)\ntoward Self-direction: action"),
    ],
    loc="lower center",
    frameon=True,
    fontsize=8.5,
    handlelength=2,
)

ax.set_title("Schwartz Value Circle", fontsize=13, fontweight="bold", pad=14)

plt.tight_layout()
plt.savefig("schwartz_value_circle.png", dpi=150,
            bbox_inches="tight")
plt.show()
print("Saved to schwartz_value_circle.png")