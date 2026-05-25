"""
Publication-ready 4-panel figure replacing Table 1.
Outputs both PDF (for LaTeX) and PNG (for preview).
"""
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

# ---- Style: aligned with visualize.py ----
mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "axes.linewidth": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# ---- Data from Table 1 (Qwen3.5-9B) ----
# Ordered: distribution-driven first (sorted by rho_T base, descending), then behavior-centric, then raw
methods = ['CAA', 'SphericalSteer', 'SAS', 'ODESteer', 'BiPO', 'OPT', 'Cold-Steer', 'Raw act.']
paradigm = ['dist','dist','dist','dist','behav','behav','behav','raw']

rhoT_b  = [0.4606, 0.3962, 0.3290, 0.2730, 0.1188, 0.1138, 0.0265, 0.2228]
rhoT_i  = [0.2351, 0.2048, 0.1526, 0.2055,-0.0104, 0.0615,-0.0568, 0.1253]
rhoT_bp = [2.3e-11,1.5e-8, 3.6e-6, 1.4e-4, 1.0e-1, 1.2e-1, 7.2e-1, 2.0e-3]
rhoT_ip = [1.1e-3, 4.6e-3, 3.6e-2, 3.7e-3, 8.9e-1, 4.0e-1, 4.4e-1, 8.5e-2]

rT_b  = [0.4754, 0.4061, 0.3326, 0.2963, 0.1291, 0.1104,-0.0100, 0.2087]
rT_i  = [0.2527, 0.2170, 0.1636, 0.2177,-0.0245, 0.0552,-0.0394, 0.1112]
rT_bp = [4.2e-12,6.1e-9, 2.7e-6, 3.3e-5, 7.6e-2, 1.3e-1, 8.9e-1, 3.9e-3]
rT_ip = [4.4e-4, 2.6e-3, 2.4e-2, 1.9e-3, 7.4e-1, 4.5e-1, 5.9e-1, 1.3e-1]

rhoH_b  = [0.3408, 0.2746, 0.3585, 0.2988, 0.1070, 0.0493, 0.0013, 0.2115]
rhoH_i  = [0.1248, 0.1009, 0.1831, 0.1495, 0.0378,-0.0177,-0.0268, 0.1455]
rhoH_bp = [1.5e-6, 1.3e-4, 3.8e-7, 2.8e-5, 1.4e-1, 5.0e-1, 9.9e-1, 3.4e-3]
rhoH_ip = [8.6e-2, 1.7e-1, 1.1e-2, 5.2e-3, 6.1e-1, 8.1e-1, 7.1e-1, 4.5e-2]

dpol_b = [0.3883, 0.3620, 0.3777, 0.0255, 0.0151, 0.0001, 0.0162, 0.0009]
dpol_i = [0.2603, 0.2397, 0.3398, 0.0176,-0.0016,-0.0099, 0.1401, 0.0004]

# ---- Colors (aligned with visualize.py palette) ----
C_DIST_B  = '#185FA5'   # distribution-driven base — matches visualize.py
C_DIST_I  = '#8BBBE5'   # distribution-driven instruct (lighter)
C_BEHAV_B = '#D85A30'   # behavior-centric base — matches visualize.py
C_BEHAV_I = '#F0A98A'   # behavior-centric instruct (lighter)
C_RAW_B   = '#5f5e5a'   # gray
C_RAW_I   = '#bdbcb7'   # light gray

def colors_for(regime):
    out = []
    for p in paradigm:
        if p == 'dist':  out.append(C_DIST_B if regime=='b' else C_DIST_I)
        elif p == 'behav': out.append(C_BEHAV_B if regime=='b' else C_BEHAV_I)
        else: out.append(C_RAW_B if regime=='b' else C_RAW_I)
    return out

def sig_marker(p):
    """Return significance asterisk."""
    if p is None: return ''
    if p < 1e-3: return '***'
    if p < 1e-2: return '**'
    if p < 5e-2: return '*'
    return ''

def plot_panel(ax, base_vals, inst_vals, base_p, inst_p, title, ylim):
    n = len(methods)
    x = np.arange(n)
    w = 0.38

    bars_b = ax.bar(x - w/2, base_vals, w, color=colors_for('b'),
                    edgecolor='black', linewidth=0.4, label='Base')
    bars_i = ax.bar(x + w/2, inst_vals, w, color=colors_for('i'),
                    edgecolor='black', linewidth=0.4, label='Instruct')

    # Significance markers above bars
    if base_p is not None:
        for xi, v, p in zip(x - w/2, base_vals, base_p):
            m = sig_marker(p)
            if m:
                ax.text(xi, v + 0.008 if v >= 0 else v - 0.025, m,
                        ha='center', va='bottom' if v>=0 else 'top',
                        fontsize=6, color='black')
    if inst_p is not None:
        for xi, v, p in zip(x + w/2, inst_vals, inst_p):
            m = sig_marker(p)
            if m:
                ax.text(xi, v + 0.008 if v >= 0 else v - 0.025, m,
                        ha='center', va='bottom' if v>=0 else 'top',
                        fontsize=6, color='black')

    ax.axhline(0, color='black', linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=35, ha='right')
    ax.set_ylim(ylim)
    ax.set_title(title, pad=4)
    ax.grid(axis='y', color='#e0e0e0', linewidth=0.7, linestyle='--')
    ax.set_axisbelow(True)

    # Visual separator between paradigm groups
    # dist (0-3), behav (4-6), raw (7)
    for boundary in [3.5, 6.5]:
        ax.axvline(boundary, color='black', linewidth=0.3, linestyle=':', alpha=0.4)

# ---- Build figure ----
fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.0))

plot_panel(axes[0,0], rhoT_b, rhoT_i, rhoT_bp, rhoT_ip,
           r'(a) Theory rank correlation $\rho_T$', (-0.1, 0.55))
plot_panel(axes[0,1], rT_b, rT_i, rT_bp, rT_ip,
           r'(b) Theory linear correlation $r_T$', (-0.1, 0.55))
plot_panel(axes[1,0], rhoH_b, rhoH_i, rhoH_bp, rhoH_ip,
           r'(c) Hierarchical correlation $\rho_H$', (-0.1, 0.45))
plot_panel(axes[1,1], dpol_b, dpol_i, None, None,
           r'(d) Polarity separation $\Delta_{pol}$', (-0.05, 0.45))

# ---- Single shared legend at top ----
from matplotlib.patches import Patch
legend_handles = [
    Patch(facecolor=C_DIST_B,  edgecolor='black', linewidth=0.4, label='Distribution-driven (Base)'),
    Patch(facecolor=C_DIST_I,  edgecolor='black', linewidth=0.4, label='Distribution-driven (Instruct)'),
    Patch(facecolor=C_BEHAV_B, edgecolor='black', linewidth=0.4, label='Behavior-centric (Base)'),
    Patch(facecolor=C_BEHAV_I, edgecolor='black', linewidth=0.4, label='Behavior-centric (Instruct)'),
    Patch(facecolor=C_RAW_B,   edgecolor='black', linewidth=0.4, label='Raw activation (Base)'),
    Patch(facecolor=C_RAW_I,   edgecolor='black', linewidth=0.4, label='Raw activation (Instruct)'),
]
fig.legend(handles=legend_handles, loc='upper center',
           bbox_to_anchor=(0.5, 1.005), ncol=3, frameon=False,
           handlelength=1.2, handleheight=0.9, columnspacing=1.4)

# Significance footnote
fig.text(0.5, -0.005,
         r'Significance: $^{***}p<10^{-3}$, $^{**}p<10^{-2}$, $^{*}p<0.05$. ',
        #  r'$\Delta_{pol}$ is a direct separation score; no p-value reported.',
         ha='center', fontsize=7, color='#444')

plt.tight_layout(rect=[0, 0.01, 1, 0.88])

# Save both PDF (for LaTeX) and PNG (for preview)
fig.savefig('table1_figure.pdf', bbox_inches='tight', pad_inches=0.05)
fig.savefig('table1_figure.png', bbox_inches='tight', pad_inches=0.05, dpi=300)
print("Saved table1_figure.pdf and table1_figure.png")
