import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

methods = [
    {'label': 'Cold-Steer', 'rhoT': 0.0265, 'gain': 5.50,  'type': 'behavior'},
    {'label': 'SAS',         'rhoT': 0.5164, 'gain': 16.2,  'type': 'distribution'},
    {'label': 'BiPO',        'rhoT': 0.1188, 'gain': 8.87,  'type': 'behavior'},
    {'label': 'SphericalSteer', 'rhoT': 0.3962, 'gain': 9.38, 'type': 'distribution'},
    {'label': 'OPT',         'rhoT': 0.0231, 'gain': 9.60,  'type': 'behavior'},
    {'label': 'ODESteer',    'rhoT': 0.2730, 'gain': 9.70,  'type': 'distribution'},
    {'label': 'CAA',         'rhoT': 0.4606, 'gain': 11.74, 'type': 'distribution'},
]

color_map = {
    'distribution': '#185FA5',
    'behavior':     '#D85A30',
}

# label offsets (x, y) to avoid overlap
offsets = {
    'Cold-Steer':    ( 0.012,  0.5),
    'SAS':           ( 0.012, -1.2),
    'BiPO':          ( 0.012,  0.5),
    'SphericalSteer':( 0.012,  0.5),
    'OPT':           (-0.055, -1.2),
    'ODESteer':      ( 0.012,  0.5),
    'CAA':           ( 0.012,  0.5),
}

fig, ax = plt.subplots(figsize=(7, 5))

for m in methods:
    c = color_map[m['type']]
    ax.scatter(
        m['rhoT'], m['gain'],
        color=c,
        edgecolors=c,
        facecolors='white' if m['type'] == 'behavior' else c,
        s=110,
        linewidths=2,
        zorder=3,
    )
    dx, dy = offsets[m['label']]
    ax.text(
        m['rhoT'] + dx,
        m['gain'] + dy,
        m['label'],
        fontsize=15,   # increased from 10
        color=c,
        va='center',
    )

# light grid
ax.set_axisbelow(True)
ax.grid(color='#e0e0e0', linewidth=0.7, linestyle='--')
ax.spines[['top', 'right']].set_visible(False)

ax.set_xlabel(r'Geometric fidelity  $\rho_T$', fontsize=16)  # increased
ax.set_ylabel('Accuracy gain', fontsize=16)                  # increased
ax.set_xlim(-0.05, 0.58)
ax.set_ylim(0, 22)

# bigger tick labels
ax.tick_params(axis='both', labelsize=13)

ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'+{v:.0f}'))

# legend
legend_handles = [
    mpatches.Patch(facecolor='#185FA5', label='Distribution-driven'),
    mpatches.Patch(facecolor='#D85A30', label='Behavior-centric'),
]
ax.legend(handles=legend_handles, fontsize=16, frameon=False, loc='upper left')

plt.tight_layout()
plt.savefig('geometry_vs_accuracy.pdf', bbox_inches='tight', dpi=300)
plt.savefig('geometry_vs_accuracy.png', bbox_inches='tight', dpi=300)
plt.show()