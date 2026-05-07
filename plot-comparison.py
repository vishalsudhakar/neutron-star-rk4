'''
plot_interpolation_comparison.py

Compares the analytical AV14+UVII solution against:
    - Cubic and linear interpolations of tabulated AV14+UVII samples (.npy)
    - APR  computed with cubic and linear interpolation (via EOSNeutronStars)
    - RG   computed with cubic and linear interpolation (via EOSNeutronStars)

Data required (relative to this script's directory):
    tabulated-plot-data/cubic_interp_test/cd.npy
    tabulated-plot-data/cubic_interp_test/m.npy
    tabulated-plot-data/cubic_interp_test/r.npy

    tabulated-plot-data/linear_interp_test/cd.npy
    tabulated-plot-data/linear_interp_test/m.npy
    tabulated-plot-data/linear_interp_test/r.npy

Plots produced (combined into one 1x3 figure):
    (a) Mass vs. central baryon density
    (b) Radius vs. central baryon density
    (c) Mass vs. radius

Saved as: interp_comparison.pdf

Usage:
    python plot_interpolation_comparison.py
'''

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib import rcParams

from EOSNeutronStars import run

# ---------------------------------------------------------------------------
# Paths to saved .npy data
# ---------------------------------------------------------------------------

_HERE       = os.path.dirname(os.path.abspath(__file__))
_CUBIC_DIR  = os.path.join(_HERE, 'tabulated-plot-data', 'cubic_interp_test')
_LINEAR_DIR = os.path.join(_HERE, 'tabulated-plot-data', 'linear_interp_test')

# ---------------------------------------------------------------------------
# Load AV14+UVII tabulated-interpolation results from disk
# ---------------------------------------------------------------------------

av14_cubic = {
    'cd': np.load(os.path.join(_CUBIC_DIR,  'cd.npy')),
    'm' : np.load(os.path.join(_CUBIC_DIR,  'm.npy')),
    'r' : np.load(os.path.join(_CUBIC_DIR,  'r.npy')),
}

av14_linear = {
    'cd': np.load(os.path.join(_LINEAR_DIR, 'cd.npy')),
    'm' : np.load(os.path.join(_LINEAR_DIR, 'm.npy')),
    'r' : np.load(os.path.join(_LINEAR_DIR, 'r.npy')),
}

# ---------------------------------------------------------------------------
# Compute solutions via EOSNeutronStars
#   AV14+UVII  i=1  t0=0.5  (analytic)
#   APR        i=4  t0=0.5  (cubic and linear)
#   RG         i=5  t0=0.5  (cubic and linear)
# ---------------------------------------------------------------------------

def _compute(i, t0, interp):
    '''Run the solver for EOS index i with the given interpolation method.'''
    print(f"  Computing EOS {i} ({interp})...")
    s = run(interp_type=interp)
    s.calculate(h=0.0007, R0=0.0, t0=t0, u0=0.0, i=i)
    print(f"    done — {len(s.cd)} density points")
    return {'cd': s.cd, 'm': s.m, 'r': s.r}

print("=" * 50)
print("  Computing all solutions")
print("=" * 50)

analytic    = _compute(i=1, t0=0.5,   interp='cubic')   # interp unused for analytic EOS
apr_cubic   = _compute(i=4, t0=0.5,   interp='cubic')
apr_linear  = _compute(i=4, t0=0.5,   interp='linear')
rg_cubic    = _compute(i=5, t0=0.5,   interp='cubic')
rg_linear   = _compute(i=5, t0=0.5,   interp='linear')

print("\nAll solutions ready. Generating plot...\n")

# ---------------------------------------------------------------------------
# Publication rcParams  (no LaTeX installation required)
# ---------------------------------------------------------------------------

rcParams.update({
    'text.usetex'        : False,
    'font.family'        : 'serif',
    'font.serif'         : ['DejaVu Serif'],
    'mathtext.fontset'   : 'cm',
    'font.size'          : 11,

    'axes.linewidth'     : 1.0,
    'axes.labelsize'     : 13,
    'axes.labelpad'      : 6,

    'xtick.direction'    : 'in',
    'ytick.direction'    : 'in',
    'xtick.top'          : True,
    'ytick.right'        : True,
    'xtick.major.size'   : 5,
    'xtick.minor.size'   : 3,
    'ytick.major.size'   : 5,
    'ytick.minor.size'   : 3,
    'xtick.major.width'  : 1.0,
    'xtick.minor.width'  : 0.8,
    'ytick.major.width'  : 1.0,
    'ytick.minor.width'  : 0.8,
    'xtick.labelsize'    : 11,
    'ytick.labelsize'    : 11,

    'lines.linewidth'    : 1.6,

    'legend.frameon'     : False,
    'legend.fontsize'    : 9.5,
    'legend.handlelength': 2.2,

    'savefig.dpi'        : 300,
    'savefig.bbox'       : 'tight',
    'savefig.pad_inches' : 0.05,
    'figure.dpi'         : 150,
})

# ---------------------------------------------------------------------------
# Curve definitions
#
# Colour encodes the EOS family:
#   AV14+UVII  — blue family
#   APR        — orange family
#   RG         — green family
#
# Linestyle encodes the method:
#   analytic   — solid
#   cubic      — dashed
#   linear     — dotted
# ---------------------------------------------------------------------------

CURVES = [
    # AV14+UVII
    {
        'data'  : analytic,
        'label' : 'AV14+UVII (analytic)',
        'color' : '#2E5FA3',
        'ls'    : '-',
        'lw'    : 2.0,
        'zorder': 5,
    },
    {
        'data'  : av14_cubic,
        'label' : 'AV14+UVII (cubic)',
        'color' : '#6A9FD8',
        'ls'    : '--',
        'lw'    : 1.6,
        'zorder': 4,
    },
    {
        'data'  : av14_linear,
        'label' : 'AV14+UVII (linear)',
        'color' : '#A8C8F0',
        'ls'    : ':',
        'lw'    : 1.6,
        'zorder': 3,
    },
    # APR
    {
        'data'  : apr_cubic,
        'label' : 'APR (cubic)',
        'color' : '#D95F02',
        'ls'    : '--',
        'lw'    : 1.6,
        'zorder': 4,
    },
    {
        'data'  : apr_linear,
        'label' : 'APR (linear)',
        'color' : '#F5A86A',
        'ls'    : ':',
        'lw'    : 1.6,
        'zorder': 3,
    },
    # RG
    {
        'data'  : rg_cubic,
        'label' : 'RG (cubic)',
        'color' : '#1B8A4A',
        'ls'    : '--',
        'lw'    : 1.6,
        'zorder': 4,
    },
    {
        'data'  : rg_linear,
        'label' : 'RG (linear)',
        'color' : '#74C99A',
        'ls'    : ':',
        'lw'    : 1.6,
        'zorder': 3,
    },
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _legend_handles():
    return [
        mlines.Line2D([], [], color=c['color'], linestyle=c['ls'],
                      linewidth=c['lw'], label=c['label'])
        for c in CURVES
    ]

def _apply_minor_ticks(ax):
    ax.minorticks_on()

# ---------------------------------------------------------------------------
# Single figure — three panels side by side
# ---------------------------------------------------------------------------

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

# Panel labels (a), (b), (c)
for ax, letter in zip((ax1, ax2, ax3), ('(a)', '(b)', '(c)')):
    ax.text(0.04, 0.96, letter, transform=ax.transAxes,
            fontsize=11, va='top', ha='left')

# --- Panel (a): Mass vs. central baryon density ---
for c in CURVES:
    ax1.plot(c['data']['cd'], c['data']['m'],
             color=c['color'], linestyle=c['ls'],
             linewidth=c['lw'], zorder=c['zorder'])

ax1.set_xlabel(r'$n_0 \ [\mathregular{fm}^{-3}]$')
ax1.set_ylabel(r'$M \ [M_{\odot}]$')
ax1.set_xlim([0, 2.5])
ax1.set_ylim([0, 3])
_apply_minor_ticks(ax1)

# --- Panel (b): Radius vs. central baryon density ---
for c in CURVES:
    ax2.plot(c['data']['cd'], c['data']['r'],
             color=c['color'], linestyle=c['ls'],
             linewidth=c['lw'], zorder=c['zorder'])

ax2.set_xlabel(r'$n_0 \ [\mathregular{fm}^{-3}]$')
ax2.set_ylabel(r'$R \ [\mathregular{km}]$')
ax2.set_xlim([0, 2.5])
ax2.set_ylim([0, 15])
_apply_minor_ticks(ax2)

# --- Panel (c): Mass vs. radius ---
for c in CURVES:
    ax3.plot(c['data']['r'], c['data']['m'],
             color=c['color'], linestyle=c['ls'],
             linewidth=c['lw'], zorder=c['zorder'])

ax3.set_xlabel(r'$R \ [\mathregular{km}]$')
ax3.set_ylabel(r'$M \ [M_{\odot}]$')
ax3.set_xlim([6, 20])
ax3.set_ylim([0, 3])
_apply_minor_ticks(ax3)

# Shared legend — two rows below all panels so seven entries don't crowd
fig.legend(
    handles=_legend_handles(),
    loc='lower center',
    ncol=4,
    bbox_to_anchor=(0.5, -0.12),
    columnspacing=1.8,
    handlelength=2.5,
)

fig.tight_layout()
fig.savefig('interp_comparison.pdf', bbox_inches='tight')
print("Saved: interp_comparison.pdf")

# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

plt.show()
