import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch


HIGHER_ORDER_GROUPS = {
    "Openness to Change": [
        "Self-direction: thought", "Self-direction: action", "Stimulation", "Hedonism",
    ],
    "Self-Enhancement": [
        "Achievement", "Power: dominance", "Power: resources", "Face",
    ],
    "Conservation": [
        "Security: personal", "Security: societal", "Tradition",
        "Conformity: rules", "Conformity: interpersonal", "Humility",
    ],
    "Self-Transcendence": [
        "Benevolence: dependability", "Benevolence: caring",
        "Universalism: concern", "Universalism: nature",
        "Universalism: tolerance", "Universalism: objectivity",
    ],
}

GROUP_COLORS = {
    "Openness to Change": "#D4A017",
    "Self-Enhancement":   "#F44336",
    "Conservation":       "#1E88E5",
    "Self-Transcendence": "#4CAF50",
}


def value_to_group(value):
    for group, members in HIGHER_ORDER_GROUPS.items():
        if value in members:
            return group
    return "Unknown"


# Alternative trend: values dip in the middle, then rebound on the right.
DISPLAY_ITEMS = [
    ("Power: dominance", "POW", 0.85, 0.0),
    ("Achievement", "ACH", 0.58, 1.0),
    ("Hedonism", "HED", 0.25, 2.0),
    ("Security: personal", "SEC", -0.20, 3.6),
    ("Universalism: concern", "UNIV", -0.33, 4.6),
    ("Benevolence: caring", "BEN", -0.44, 5.6),
]

values = [item[0] for item in DISPLAY_ITEMS]
labels = [item[1] for item in DISPLAY_ITEMS]
heights = [item[2] for item in DISPLAY_ITEMS]
x = np.array([item[3] for item in DISPLAY_ITEMS])
colors = [GROUP_COLORS[value_to_group(v)] for v in values]

mpl.rcParams.update({
    "font.family": "sans-serif",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "pdf.fonttype": 42,
})

fig, ax = plt.subplots(figsize=(10, 4))
ax.bar(x, heights, color=colors, width=0.65, edgecolor="none", zorder=3)

ax.axhline(0, color="black", linewidth=0.8, zorder=2)
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=16, fontweight="bold")
for tick, color in zip(ax.get_xticklabels(), colors):
    tick.set_color(color)

ax.tick_params(axis="x", length=0, pad=6)
ax.tick_params(axis="y", labelsize=11)
ax.set_ylabel("Transfer strength", fontsize=16)
ax.set_xlim(-0.6, 6.2)
ax.set_ylim(-1.15, 1.1)
ax.grid(axis="y", color="#e0e0e0", linewidth=0.6, linestyle="--", zorder=1)
ax.set_axisbelow(True)

legend_handles = [
    Patch(facecolor=GROUP_COLORS[group], label=group) for group in HIGHER_ORDER_GROUPS
]
# ax.legend(handles=legend_handles, loc="upper right", frameon=False, fontsize=16, ncol=2)

plt.tight_layout()
plt.savefig("transfer_strength_alt.pdf", bbox_inches="tight", pad_inches=0.05)
plt.savefig("transfer_strength_alt.png", bbox_inches="tight", pad_inches=0.05, dpi=300)
plt.show()
print("Saved transfer_strength_alt.pdf / .png")
