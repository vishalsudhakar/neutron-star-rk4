'''
plot_all_eos.py

Generates all three standard plots for every supported Equation of State:

    i = 1  ->  AV14+UVII  (analytic)
    i = 2  ->  UV14-UVII  (analytic)
    i = 3  ->  UV14-TN1   (analytic)
    i = 4  ->  APR        (tabulated, cubic spline)
    i = 5  ->  RG         (tabulated, cubic spline)

Plots produced:
    1. Mass vs. central baryon density
    2. Radius vs. central baryon density
    3. Mass vs. radius  (with PSR J0030+0451 and PSR J0740+6620 constraints)

Each EOS is computed once via calculate() and then drawn onto each figure.
All three figures are saved as PDFs and displayed at the end.

Usage:
    python plot_all_eos.py

Initial conditions (per the solver documentation):
    h  = 0.01
    R0 = 0
    u0 = 0
    t0 = 0.5   (UV14-TN1 uses t0 = 0.829)
'''

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib import rcParams

from EOSNeutronStars import run

# ---------------------------------------------------------------------------
# EOS catalogue
# Each entry: (index, label, t0, colour, linestyle)
# Analytic EOS use solid lines; tabulated EOS use dashed to distinguish origin.
# t0 is EOS-specific per the solver documentation.
# ---------------------------------------------------------------------------

EOS_CATALOGUE = [
    (1, 'AV14+UVII', 0.5,   '#2E5FA3', '-' ),   # steel blue   — analytic
    (2, 'UV14-UVII', 0.5,   '#D95F02', '-' ),   # burnt orange — analytic
    (3, 'UV14-TN1',  0.829, '#1B8A4A', '-' ),   # forest green — analytic
    (4, 'APR',       0.5,   '#7B2D8B', '--'),   # purple       — tabulated
    (5, 'RG',        0.5,   '#B22222', '--'),   # firebrick    — tabulated
]

# Shared initial conditions
H  = 0.0007
R0 = 0.0
U0 = 0.0

# ---------------------------------------------------------------------------
# Compute all EOS
# ---------------------------------------------------------------------------

print("=" * 55)
print("  Neutron Star Mass-Radius — computing all EOS")
print("=" * 55)

solvers = {}

for (i, label, t0, color, ls) in EOS_CATALOGUE:
    print(f"\n--- {label} (EOS {i}) ---")
    solver = run(interp_type='cubic')
    solver.calculate(H, R0, t0, U0, i)
    solvers[i] = (solver, label, color, ls)
    print(f"    done — {len(solver.cd)} density points computed")

print("\nAll EOS computed. Generating plots...\n")

# ---------------------------------------------------------------------------
# Publication rcParams
# Targets a single-column journal figure (7 x 5 in).
# Uses LaTeX / Computer Modern for typography consistent with a physics
# manuscript. Ticks are inward on all four sides with minor ticks enabled.
# ---------------------------------------------------------------------------

rcParams.update({
    # Font — uses matplotlib's built-in math renderer (no LaTeX installation required).
    # Switch text.usetex to True and set font.serif to ['Computer Modern Roman']
    # if a LaTeX distribution (MacTeX / texlive) is available for native-quality output.
    'text.usetex'        : False,
    'font.family'        : 'serif',
    'font.serif'         : ['DejaVu Serif'],
    'mathtext.fontset'   : 'cm',
    'font.size'          : 11,

    # Axes
    'axes.linewidth'      : 1.0,
    'axes.labelsize'      : 13,
    'axes.labelpad'       : 6,

    # Ticks — inward on all four sides
    'xtick.direction'     : 'in',
    'ytick.direction'     : 'in',
    'xtick.top'           : True,
    'ytick.right'         : True,
    'xtick.major.size'    : 5,
    'xtick.minor.size'    : 3,
    'ytick.major.size'    : 5,
    'ytick.minor.size'    : 3,
    'xtick.major.width'   : 1.0,
    'xtick.minor.width'   : 0.8,
    'ytick.major.width'   : 1.0,
    'ytick.minor.width'   : 0.8,
    'xtick.labelsize'     : 11,
    'ytick.labelsize'     : 11,

    # Lines
    'lines.linewidth'     : 1.6,

    # Legend
    'legend.frameon'      : False,
    'legend.fontsize'     : 10,
    'legend.handlelength' : 2.2,

    # Output
    'savefig.dpi'         : 300,
    'savefig.bbox'        : 'tight',
    'savefig.pad_inches'  : 0.05,
    'figure.dpi'          : 150,
})

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _apply_minor_ticks(ax):
    '''Enable minor ticks on both axes.'''
    ax.minorticks_on()


def _eos_legend_handles(solvers):
    '''
    Build Line2D legend handles from the solver catalogue.
    Solid lines for analytic EOS, dashed for tabulated.
    '''
    return [
        mlines.Line2D([], [], color=color, linestyle=ls,
                      linewidth=1.6, label=label)
        for _, (_, label, color, ls) in solvers.items()
    ]


# ---------------------------------------------------------------------------
# Figure 1 — Mass vs. central baryon density
# ---------------------------------------------------------------------------

fig1, ax1 = plt.subplots(figsize=(7, 5))

for i, (solver, label, color, ls) in solvers.items():
    ax1.plot(solver.cd, solver.m, color=color, linestyle=ls)

ax1.set_xlabel(r'$n_0 \ (\mathrm{fm}^{-3})$')
ax1.set_ylabel(r'$M \ (M_{\odot})$')
ax1.set_xlim([0, 2.5])
ax1.set_ylim([0, 3])
ax1.legend(handles=_eos_legend_handles(solvers), loc='lower right')
_apply_minor_ticks(ax1)
fig1.tight_layout()
fig1.savefig('mass_n0_all.pdf')
print("Saved: mass_n0_all.pdf")

# ---------------------------------------------------------------------------
# Figure 2 — Radius vs. central baryon density
# ---------------------------------------------------------------------------

fig2, ax2 = plt.subplots(figsize=(7, 5))

for i, (solver, label, color, ls) in solvers.items():
    ax2.plot(solver.cd, solver.r, color=color, linestyle=ls)

ax2.set_xlabel(r'$n_0 \ (\mathrm{fm}^{-3})$')
ax2.set_ylabel(r'$R \ (\mathrm{km})$')
ax2.set_xlim([0, 2.5])
ax2.set_ylim([0, 15])
ax2.legend(handles=_eos_legend_handles(solvers), loc='lower right')
_apply_minor_ticks(ax2)
fig2.tight_layout()
fig2.savefig('radius_n0_all.pdf')
print("Saved: radius_n0_all.pdf")

# ---------------------------------------------------------------------------
# Figure 3 — Mass vs. radius  (+ observational constraints)
# ---------------------------------------------------------------------------

fig3, ax3 = plt.subplots(figsize=(7, 5))

for i, (solver, label, color, ls) in solvers.items():
    ax3.plot(solver.r, solver.m, color=color, linestyle=ls)

# Observational constraints — NICER + XMM-Newton results
obs = [
    {
        'label'  : r'PSR J0030$+$0451',
        'x'      : 13.02, 'y'    : 1.44,
        'xerr'   : [[1.06], [1.24]],
        'yerr'   : [[0.14], [0.07]],
        'offset' : (8, 8),
    },
    {
        'label'  : r'PSR J0740$+$6620',
        'x'      : 13.7,  'y'    : 2.08,
        'xerr'   : [[1.5],  [2.6]],
        'yerr'   : [[0.15], [0.07]],
        'offset' : (8, -16),
    },
]

for o in obs:
    ax3.errorbar(
        o['x'], o['y'],
        xerr=o['xerr'], yerr=o['yerr'],
        fmt='o', color='#333333',
        markersize=4, markeredgewidth=0.8,
        elinewidth=1.0, capsize=3, capthick=1.0,
        zorder=5,
    )
    ax3.annotate(
        o['label'], (o['x'], o['y']),
        textcoords='offset points', xytext=o['offset'],
        fontsize=9.5,
    )

ax3.set_xlabel(r'$R \ (\mathrm{km})$')
ax3.set_ylabel(r'$M \ (M_{\odot})$')
ax3.set_xlim([6, 20])
ax3.set_ylim([0, 3])

obs_handle = mlines.Line2D(
    [], [], color='#333333', marker='o', linestyle='None',
    markersize=4, label='NICER observations',
)
ax3.legend(
    handles=_eos_legend_handles(solvers) + [obs_handle],
    loc='lower right',
)
_apply_minor_ticks(ax3)
fig3.tight_layout()
fig3.savefig('mass_radius_all.pdf')
print("Saved: mass_radius_all.pdf")

# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

plt.show()
