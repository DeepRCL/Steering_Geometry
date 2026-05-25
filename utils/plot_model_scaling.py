"""
Table 2 as a chart: model size (x) vs geometric fidelity rho_T (y),
one color per model family. Matches Figures 3 and 4 visual style.
"""
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 13,
    "axes.labelsize": 14,
    "axes.titlesize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "axes.linewidth": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# ---- Data from Table 2 ----
# (params_B, rho_T, model_name_short)
families = {
    'Qwen3.5':  [(0.8, 0.2537, '0.8B'), (2, 0.3364, '2B'), (4, 0.4109, '4B'), (9, 0.4599, '9B')],
    'Qwen2.5':  [(7, 0.2370, '7B'),    (14, 0.3133, '14B'), (32, 0.3347, '32B')],
    'Gemma-4':  [(31, 0.2930, '31B')],
    'Gemma-3':  [(1, 0.1270, '1B'),    (4, 0.2398, '4B'),  (12, 0.1611, '12B')],
    'Llama-3.1':[(8, 0.3701, '8B')],
    'Mistral':  [(7, 0.3248, '7B')],
}

# Colors: aligned with visualize.py palette
colors = {
    'Qwen3.5':   '#185FA5',  # primary distribution blue — matches visualize.py
    'Qwen2.5':   '#6BADD6',  # lighter blue
    'Gemma-4':   '#D85A30',  # primary behavior coral — matches visualize.py
    'Gemma-3':   '#F0A98A',  # lighter coral
    'Llama-3.1': '#7030a0',  # purple
    'Mistral':   '#548235',  # green
}

markers = {
    'Qwen3.5':   'o',
    'Qwen2.5':   's',
    'Gemma-4':   'D',
    'Gemma-3':   '^',
    'Llama-3.1': 'P',
    'Mistral':   'X',
}

fig, ax = plt.subplots(figsize=(6.5, 4.2))

# Plot multi-point families as lines + markers
for fam in ['Qwen3.5', 'Qwen2.5', 'Gemma-3']:
    pts = families[fam]
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    ax.plot(xs, ys, color=colors[fam], linewidth=1.4, marker=markers[fam],
            markersize=8, markeredgecolor='black', markeredgewidth=0.5,
            label=fam, zorder=3)

# Plot singletons as markers only
for fam in ['Gemma-4', 'Llama-3.1', 'Mistral']:
    pts = families[fam]
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    ax.scatter(xs, ys, color=colors[fam], marker=markers[fam],
               s=80, edgecolor='black', linewidth=0.5, label=fam, zorder=3)

# Annotate each point with model size label
all_pts = []
for fam, pts in families.items():
    for x, y, label in pts:
        all_pts.append((fam, x, y, label))

# Manual offsets to avoid overlap
offsets = {
    ('Qwen3.5','0.8B'):  (-8, -14),
    ('Qwen3.5','2B'):    (6, -4),
    ('Qwen3.5','4B'):    (6, -4),
    ('Qwen3.5','9B'):    (6, 4),
    ('Qwen2.5','7B'):    (-6, -14),
    ('Qwen2.5','14B'):   (6, -4),
    ('Qwen2.5','32B'):   (6, 4),
    ('Gemma-3','1B'):    (6, -4),
    ('Gemma-3','4B'):    (6, 4),
    ('Gemma-3','12B'):   (6, -4),
    ('Gemma-4','31B'):   (-10, -14),
    ('Llama-3.1','8B'):  (6, 4),
    ('Mistral','7B'):    (6, 4),
}

for fam, x, y, label in all_pts:
    dx, dy = offsets.get((fam, label), (6, 4))
    ax.annotate(label, xy=(x, y), xytext=(dx, dy), textcoords='offset points',
                fontsize=10, color='#333')

ax.set_xscale('log')
ax.set_xticks([1, 2, 4, 8, 16, 32])
ax.set_xticklabels(['1B', '2B', '4B', '8B', '16B', '32B'])
ax.set_xlim(0.6, 50)
ax.set_ylim(0.05, 0.52)

ax.set_xlabel('Model size (parameters, log scale)')
ax.set_ylabel(r'Geometric fidelity $\rho_T$')

ax.grid(True, color='#e0e0e0', linewidth=0.7, linestyle='--')
ax.set_axisbelow(True)

# Legend ordered to put multi-point families first
handles, labels = ax.get_legend_handles_labels()
order = ['Qwen3.5', 'Qwen2.5', 'Gemma-4', 'Gemma-3', 'Llama-3.1', 'Mistral']
handle_dict = dict(zip(labels, handles))
ordered_handles = [handle_dict[l] for l in order if l in handle_dict]
ax.legend(ordered_handles, order, loc='lower right', frameon=False,
          ncol=2, handletextpad=0.4, columnspacing=1.0)

plt.tight_layout()
fig.savefig('table2_figure.pdf', bbox_inches='tight', pad_inches=0.05)
fig.savefig('table2_figure.png', bbox_inches='tight', pad_inches=0.05, dpi=300)
print("Saved table2_figure.pdf and table2_figure.png")
