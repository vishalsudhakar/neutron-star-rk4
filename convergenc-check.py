'''
convergence_study.py

Performs a convergence study of the RK4 solver by running the AV14+UVII EOS
(i=1) at progressively finer step sizes and examining how the mass-radius
curve and key scalar outputs (maximum mass, corresponding radius) change.

Three outputs are produced:

    Figure 1 — Mass-radius curves for all step sizes overlaid on one axes,
               allowing visual confirmation of curve convergence.

    Figure 2 — Fractional change in maximum mass and its corresponding radius
               between successive step sizes:
                   |M(h) - M(h/10)| / M(h/10)
               Plotted against h on a log-log scale. A horizontal dashed line
               marks the 0.1% convergence threshold.

    Table     — Printed to stdout: h, M_max, R(M_max), and the fractional
               changes between successive runs.

Saved PDFs:
    convergence_mass_radius.pdf
    convergence_fractional_change.pdf

Usage:
    python convergence_study.py

Initial conditions (AV14+UVII):
    R0 = 0, t0 = 0.5, u0 = 0, i = 1
'''

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

from EOSNeutronStars import run

# ---------------------------------------------------------------------------
# Study parameters
# ---------------------------------------------------------------------------

STEP_SIZES = [0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001]   # h values, coarse → fine

R0 = 0.0
T0 = 0.5
U0 = 0.0
I   = 1      # AV14+UVII

# Convergence threshold for the fractional-change plot annotation
THRESHOLD = 0.001   # 0.1 %

# ---------------------------------------------------------------------------
# Colour map — one colour per step size, dark → light as h decreases
# (coarse runs are dark, fine runs are light so the converged curve is clear)
# ---------------------------------------------------------------------------

COLOURS = {
    0.0500 : '#1A1A2E',   # Deep Navy (Base)
    0.0100 : '#253B5E',   # Dark Steel Blue
    0.0050 : '#326092',   # Subdued Ocean Blue
    0.0010 : '#5088B9',   # Classic Mid Blue
    0.0005 : '#82B1D3',   # Soft Sky Blue
    0.0001 : '#B0D4EC',   # Pale Ice Blue
}

# ---------------------------------------------------------------------------
# Publication rcParams
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
    'legend.fontsize'    : 10,
    'legend.handlelength': 2.2,

    'savefig.dpi'        : 300,
    'savefig.bbox'       : 'tight',
    'savefig.pad_inches' : 0.05,
    'figure.dpi'         : 150,
})

# ---------------------------------------------------------------------------
# Helper — extract maximum mass and its corresponding radius
# ---------------------------------------------------------------------------

def peak_mass_and_radius(solver):
    '''
    Extracts the maximum mass and its corresponding radius from a completed
    solver run.

    Parameters:
    solver : run — a run instance after calculate() has been called

    Returns:
    M_max : float — maximum mass [M_sun]
    R_max : float — radius at maximum mass [km]
    '''
    idx   = np.argmax(solver.m)
    M_max = solver.m[idx]
    R_max = solver.r[idx]
    return M_max, R_max

# ---------------------------------------------------------------------------
# Run solver at each step size and collect results
# ---------------------------------------------------------------------------

print("=" * 58)
print("  Convergence study — AV14+UVII (i=1)")
print("=" * 58)

results = {}   # h -> {'cd', 'r', 'm', 'M_max', 'R_max'}

for h in STEP_SIZES:
    print(f"\n  h = {h}")
    solver = run(interp_type='cubic')
    solver.calculate(h=h, R0=R0, t0=T0, u0=U0, i=I)
    M_max, R_max = peak_mass_and_radius(solver)
    results[h] = {
        'cd'   : solver.cd,
        'r'    : solver.r,
        'm'    : solver.m,
        'M_max': M_max,
        'R_max': R_max,
    }
    print(f"    M_max = {M_max:.4f} M_sun   R(M_max) = {R_max:.4f} km")

# ---------------------------------------------------------------------------
# Convergence table — printed to stdout
# ---------------------------------------------------------------------------

print("\n" + "=" * 74)
print(f"  {'h':>10}  {'M_max [M_sun]':>15}  {'R(M_max) [km]':>15}  "
      f"{'dM/M':>12}  {'dR/R':>12}")
print("=" * 74)

h_list = STEP_SIZES
for k, h in enumerate(h_list):
    M  = results[h]['M_max']
    R  = results[h]['R_max']

    if k == 0:
        # No previous step to compare against
        print(f"  {h:>10.4f}  {M:>15.6f}  {R:>15.6f}  {'—':>12}  {'—':>12}")
    else:
        h_prev  = h_list[k - 1]
        M_prev  = results[h_prev]['M_max']
        R_prev  = results[h_prev]['R_max']
        dM_frac = abs(M - M_prev) / M          # fractional change relative to finer run
        dR_frac = abs(R - R_prev) / R
        converged = " ✓" if dM_frac < THRESHOLD else ""
        print(f"  {h:>10.4f}  {M:>15.6f}  {R:>15.6f}  "
              f"{dM_frac:>12.2e}  {dR_frac:>12.2e}{converged}")

print("=" * 74)
print(f"  ✓ = fractional change in M_max below {THRESHOLD*100:.1f}% threshold\n")

# ---------------------------------------------------------------------------
# Figure 1 — Mass-radius curves for all step sizes
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Unified Figure — Mass-radius curves and Convergence Study
# ---------------------------------------------------------------------------

# Create a figure with 1 row and 2 columns
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# --- Plot (a): Mass-Radius Curves ---
for h in STEP_SIZES:
    d = results[h]
    ax1.plot(d['r'], d['m'],
             color=COLOURS[h],
             linewidth=1.6,
             label=rf'$h = {h}$')

ax1.set_xlabel(r'$R \ [\mathregular{km}]$')
ax1.set_ylabel(r'$M \ [M_{\odot}]$')
ax1.set_xlim([6, 20])
ax1.set_ylim([0, 3])
ax1.legend(loc='lower right', title='Step size', title_fontsize=10)
ax1.minorticks_on()

# Add label 'a'
ax1.text(-0.12, 1.05, '(a)', transform=ax1.transAxes, 
         fontsize=11, va='top', ha='right')

# --- Plot (b): Fractional Change ---
h_compare   = []   # coarser h of each pair
dM_frac_arr = []

for k in range(1, len(h_list)):
    h_fine   = h_list[k]
    h_coarse = h_list[k - 1]
    dM = abs(results[h_coarse]['M_max'] - results[h_fine]['M_max']) / results[h_fine]['M_max']
    h_compare.append(h_coarse)
    dM_frac_arr.append(dM)

ax2.loglog(h_compare, dM_frac_arr,
           color='#2E5FA3', marker='o', markersize=6,
           linewidth=1.6, label=r'$|\Delta M_{\mathregular{max}}| / M_{\mathregular{max}}$')

# Convergence threshold line
ax2.axhline(THRESHOLD, color='#555555', linewidth=1.0, linestyle='--')

ax2.set_xlabel(r'Step size $h$')
ax2.set_ylabel(r'Fractional change')
ax2.legend(loc='upper left')
ax2.minorticks_on()

# Add label 'b'
ax2.text(-0.12, 1.05, '(b)', transform=ax2.transAxes, 
         fontsize=11, va='top', ha='right')

# Final adjustments and saving
plt.tight_layout()
fig.savefig('convergence_combined_study.pdf')
print("Saved: convergence_combined_study.pdf")

plt.show()
