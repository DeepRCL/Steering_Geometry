import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

# ── Schwartz circumplex order and group colours (from CAA/Geometry/config.py) ─
SCHWARTZ_CIRCUMPLEX_ORDER = [
    "Self-direction: thought",
    "Self-direction: action",
    "Stimulation",
    "Hedonism",
    "Achievement",
    "Power: dominance",
    "Power: resources",
    "Face",
    "Security: personal",
    "Security: societal",
    "Tradition",
    "Conformity: rules",
    "Conformity: interpersonal",
    "Humility",
    "Benevolence: dependability",
    "Benevolence: caring",
    "Universalism: concern",
    "Universalism: nature",
    "Universalism: tolerance",
    "Universalism: objectivity",
]

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

def value_to_group(v):
    for g, members in HIGHER_ORDER_GROUPS.items():
        if v in members:
            return g
    return "Unknown"

# ── Short abbreviations matching the image x-axis labels ─────────────────────
SHORT_LABELS = {
    "Self-direction: thought":    "SELF-T",
    "Self-direction: action":     "SELF-A",
    "Stimulation":                "STIM",
    "Hedonism":                   "HED",
    "Achievement":                "ACH",
    "Power: dominance":           "POW-D",
    "Power: resources":           "POW-R",
    "Face":                       "FACE",
    "Security: personal":         "SEC-P",
    "Security: societal":         "SEC-S",
    "Tradition":                  "TRAD",
    "Conformity: rules":          "CON-R",
    "Conformity: interpersonal":  "CON-I",
    "Humility":                   "HUM",
    "Benevolence: dependability": "BEN-D",
    "Benevolence: caring":        "BEN-C",
    "Universalism: concern":      "UNIV-C",
    "Universalism: nature":       "UNIV-N",
    "Universalism: tolerance":    "UNIV-T",
    "Universalism: objectivity":  "UNIV-O",
}

# ── Replace with your actual per-value transfer-strength data ─────────────────
# Values should be in SCHWARTZ_CIRCUMPLEX_ORDER sequence.
transfer_strength = {
    "Self-direction: thought":     0.60,
    "Self-direction: action":      0.45,
    "Stimulation":                 0.30,
    "Hedonism":                    0.20,
    "Achievement":                 0.82,
    "Power: dominance":            0.68,
    "Power: resources":            0.50,
    "Face":                        0.10,
    "Security: personal":          0.02,
    "Security: societal":         -0.05,
    "Tradition":                  -0.15,
    "Conformity: rules":          -0.20,
    "Conformity: interpersonal":  -0.25,
    "Humility":                   -0.30,
    "Benevolence: dependability":  0.05,
    "Benevolence: caring":        -0.10,
    "Universalism: concern":      -0.45,
    "Universalism: nature":       -0.60,
    "Universalism: tolerance":    -0.55,
    "Universalism: objectivity":  -0.80,
}

# ── Build ordered arrays ──────────────────────────────────────────────────────
# Important values only, ordered to show the trend from +1 to -1. The explicit
# gap marks skipped intermediate values from the full Schwartz circumplex.
DISPLAY_ITEMS = [
    ("Power: dominance", "POW", 1.00, 0.0),
    ("Achievement", "ACH", 0.78, 1.0),
    ("Hedonism", "HED", 0.52, 2.0),
    ("Security: personal", "SEC", -0.45, 3.6),
    ("Universalism: concern", "UNIV", -0.70, 4.6),
    ("Benevolence: caring", "BEN", -1.00, 5.6),
]

values = [item[0] for item in DISPLAY_ITEMS]
labels = [item[1] for item in DISPLAY_ITEMS]
heights = [item[2] for item in DISPLAY_ITEMS]
x = np.array([item[3] for item in DISPLAY_ITEMS])
colors = [GROUP_COLORS[value_to_group(v)] for v in values]

# ── Plot ──────────────────────────────────────────────────────────────────────
mpl.rcParams.update({
    "font.family": "sans-serif",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "pdf.fonttype": 42,
})

fig, ax = plt.subplots(figsize=(10, 4))

bars = ax.bar(x, heights, color=colors, width=0.65, edgecolor="none", zorder=3)

ax.axhline(0, color="black", linewidth=0.8, zorder=2)
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=10, fontweight="bold")
for tick, c in zip(ax.get_xticklabels(), colors):
    tick.set_color(c)

# Dots in the gap indicate omitted middle values.
ax.text(2.8, -1.12, ". . .", ha="center", va="top", fontsize=16, fontweight="bold", color="#666666")
ax.tick_params(axis="x", length=0, pad=6)
ax.tick_params(axis="y", labelsize=11)
ax.set_ylabel("Transfer strength", fontsize=12)
ax.set_xlim(-0.6, 6.2)
ax.set_ylim(-1.15, 1.1)
ax.grid(axis="y", color="#e0e0e0", linewidth=0.6, linestyle="--", zorder=1)
ax.set_axisbelow(True)

# Group legend
from matplotlib.patches import Patch
legend_handles = [
    Patch(facecolor=GROUP_COLORS[g], label=g) for g in HIGHER_ORDER_GROUPS
]
# ax.legend(handles=legend_handles, loc="upper right", frameon=False,
#           fontsize=10, ncol=2)

plt.tight_layout()
plt.savefig("transfer_strength.pdf", bbox_inches="tight", pad_inches=0.05)
plt.savefig("transfer_strength.png", bbox_inches="tight", pad_inches=0.05, dpi=300)
plt.show()
print("Saved transfer_strength.pdf / .png")