"""Generate publication-quality figures from simulation_results.json."""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

RESULTS_PATH = Path(__file__).parent / "simulation_results.json"
FIG_DIR = Path(__file__).parent / "figures"
FIG_DIR.mkdir(exist_ok=True)

with open(RESULTS_PATH) as f:
    data = json.load(f)

METHOD_COLORS = {
    "Linear PRS": "#4C72B0",
    "Gaussian NN": "#DD8452",
    "Interaction NN": "#8172B3",
    "Oracle NN": "#937860",
}
METHOD_STYLES = {
    "Linear PRS": {"linestyle": "--", "marker": "^"},
    "Gaussian NN": {"linestyle": "-", "marker": "o"},
    "Interaction NN": {"linestyle": "-", "marker": "s"},
    "Oracle NN": {"linestyle": "-", "marker": "D"},
}
METHOD_ORDER = ["Linear PRS", "Gaussian NN", "Interaction NN", "Oracle NN"]

plt.rcParams.update({
    "figure.figsize": (10, 6),
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 13,
    "legend.fontsize": 11,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.15,
})


def group_by_method(results):
    by_method = defaultdict(list)
    for r in results:
        by_method[r["method"]].append(r["r2"])
    return by_method


# =====================================================================
# Figure 1: Main comparison bar chart (Experiment 1)
# =====================================================================
fig, ax = plt.subplots(figsize=(9, 5.5))
by_method = group_by_method(data["experiment1"])

x = np.arange(len(METHOD_ORDER))
means = [np.mean(by_method[m]) for m in METHOD_ORDER]
stds = [np.std(by_method[m]) for m in METHOD_ORDER]
ses = [s / np.sqrt(len(by_method[m])) for s, m in zip(stds, METHOD_ORDER)]
colors = [METHOD_COLORS[m] for m in METHOD_ORDER]

bars = ax.bar(x, means, yerr=ses, capsize=5, color=colors, edgecolor="white",
              linewidth=0.8, width=0.65, zorder=3)

for i, (m, v) in enumerate(zip(means, ses)):
    ax.text(i, m + v + 0.008, f"{m:.3f}", ha="center", va="bottom", fontsize=10.5,
            fontweight="bold")

ax.set_xticks(x)
ax.set_xticklabels(METHOD_ORDER, rotation=20, ha="right")
ax.set_ylabel("Prediction R²")
ax.set_title("Prediction R² by Method\n(p=30, nonlinear_frac=0.25, 10 replicates)")
ax.set_ylim(0, max(means) + 0.12)
ax.grid(axis="y", alpha=0.3, zorder=0)

fig.tight_layout()
fig.savefig(FIG_DIR / "fig1_main_comparison.png")
print(f"Saved {FIG_DIR / 'fig1_main_comparison.png'}")
plt.close(fig)


# =====================================================================
# Figure 2: Nonlinear fraction sweep (Experiment 2)
# =====================================================================
nf_data = data["experiment2"]
nonlinear_fracs = sorted(float(k) for k in nf_data.keys())

fig, ax = plt.subplots(figsize=(10, 6))

for method in METHOD_ORDER:
    method_means = []
    method_ses = []
    for nf in nonlinear_fracs:
        vals = [r["r2"] for r in nf_data[str(nf)] if r["method"] == method]
        method_means.append(np.mean(vals))
        method_ses.append(np.std(vals) / np.sqrt(len(vals)))

    style = METHOD_STYLES[method]
    ax.errorbar(nonlinear_fracs, method_means, yerr=method_ses,
                marker=style["marker"], linestyle=style["linestyle"],
                markersize=7, linewidth=2.2, capsize=4,
                label=method, color=METHOD_COLORS[method])

ax.set_xlabel("Nonlinear Fraction (Var(NL) / Var(L))")
ax.set_ylabel("Prediction R²")
ax.set_title("R² vs Nonlinear Fraction\n(p=30, n=5000, 5 replicates per point)")
ax.set_xticks(nonlinear_fracs)
ax.legend(loc="upper right", framealpha=0.9)
ax.set_ylim(0, 0.6)
ax.grid(alpha=0.3)

fig.tight_layout()
fig.savefig(FIG_DIR / "fig2_nonlinear_fraction_sweep.png")
print(f"Saved {FIG_DIR / 'fig2_nonlinear_fraction_sweep.png'}")
plt.close(fig)


# =====================================================================
# Figure 3: Sample size sweep (Experiment 3)
# =====================================================================
ns_data = data["experiment3"]
n_trains = sorted(int(k) for k in ns_data.keys())

fig, ax = plt.subplots(figsize=(10, 6))

for method in METHOD_ORDER:
    method_means = []
    method_ses = []
    for n in n_trains:
        vals = [r["r2"] for r in ns_data[str(n)] if r["method"] == method]
        method_means.append(np.mean(vals))
        method_ses.append(np.std(vals) / np.sqrt(len(vals)))

    style = METHOD_STYLES[method]
    ax.errorbar(n_trains, method_means, yerr=method_ses,
                marker=style["marker"], linestyle=style["linestyle"],
                markersize=7, linewidth=2.2, capsize=4,
                label=method, color=METHOD_COLORS[method])

ax.set_xlabel("Training Sample Size (n)")
ax.set_ylabel("Prediction R²")
ax.set_title("R² vs Sample Size\n(p=30, nonlinear_frac=0.25, 5 replicates per point)")
ax.set_xscale("log")
ax.set_xticks(n_trains)
ax.set_xticklabels([str(n) for n in n_trains])
ax.legend(loc="lower right", framealpha=0.9)
ax.set_ylim(0, 0.6)
ax.grid(alpha=0.3)

fig.tight_layout()
fig.savefig(FIG_DIR / "fig3_sample_size_sweep.png")
print(f"Saved {FIG_DIR / 'fig3_sample_size_sweep.png'}")
plt.close(fig)


# =====================================================================
# Figure 4: Gap-to-Oracle analysis (derived from Experiment 2)
# =====================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

FIG4_METHODS = [
    ("Linear PRS", METHOD_COLORS["Linear PRS"], "--", "^"),
    ("Gaussian NN", METHOD_COLORS["Gaussian NN"], "-", "o"),
    ("Interaction NN", METHOD_COLORS["Interaction NN"], "-", "s"),
]

# Left panel: absolute R² gap to Oracle
for method, color, ls, mkr in FIG4_METHODS:
    gaps = []
    gap_ses = []
    for nf in nonlinear_fracs:
        oracle_vals = [r["r2"] for r in nf_data[str(nf)] if r["method"] == "Oracle NN"]
        method_vals = [r["r2"] for r in nf_data[str(nf)] if r["method"] == method]
        per_rep_gaps = [o - m for o, m in zip(oracle_vals, method_vals)]
        gaps.append(np.mean(per_rep_gaps))
        gap_ses.append(np.std(per_rep_gaps) / np.sqrt(len(per_rep_gaps)))

    ax1.errorbar(nonlinear_fracs, gaps, yerr=gap_ses,
                 marker=mkr, linestyle=ls, markersize=7, linewidth=2.2, capsize=4,
                 label=method, color=color)

ax1.set_xlabel("Nonlinear Fraction")
ax1.set_ylabel("R² Gap to Oracle")
ax1.set_title("Gap to Oracle NN")
ax1.set_xticks(nonlinear_fracs)
ax1.legend(framealpha=0.9)
ax1.grid(alpha=0.3)

# Right panel: fraction of Oracle R² captured
for method, color, ls, mkr in FIG4_METHODS:
    fractions = []
    for nf in nonlinear_fracs:
        oracle_mean = np.mean([r["r2"] for r in nf_data[str(nf)] if r["method"] == "Oracle NN"])
        method_mean = np.mean([r["r2"] for r in nf_data[str(nf)] if r["method"] == method])
        fractions.append(method_mean / oracle_mean * 100 if oracle_mean > 0 else 0)

    ax2.plot(nonlinear_fracs, fractions,
             marker=mkr, linestyle=ls, markersize=7, linewidth=2.2,
             label=method, color=color)

ax2.set_xlabel("Nonlinear Fraction")
ax2.set_ylabel("% of Oracle R² Captured")
ax2.set_title("Fraction of Oracle Performance")
ax2.set_xticks(nonlinear_fracs)
ax2.set_ylim(0, 110)
ax2.axhline(100, color="gray", linestyle="--", alpha=0.5, linewidth=1)
ax2.legend(framealpha=0.9)
ax2.grid(alpha=0.3)

fig.tight_layout()
fig.savefig(FIG_DIR / "fig4_gap_analysis.png")
print(f"Saved {FIG_DIR / 'fig4_gap_analysis.png'}")
plt.close(fig)

print("\nAll figures generated successfully.")
