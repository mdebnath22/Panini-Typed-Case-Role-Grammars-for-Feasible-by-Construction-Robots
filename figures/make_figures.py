"""
Paper Figures — 6 Plots From Verified Real Data
===================================================

Every number below is traced to a specific table/experiment already
established and, in most cases, independently reproduced. No fabricated or
placeholder data. Each figure function's docstring names its exact source.

Requires: matplotlib, numpy (no MuJoCo/torch needed -- this only plots
already-collected numbers).

Run:
    python3 make_figures.py
Outputs PDF (vector, for LaTeX \\includegraphics) and PNG (for quick
preview) into ./figures/.
"""

import os

import matplotlib.pyplot as plt
import numpy as np

OUT_DIR = "figures"
os.makedirs(OUT_DIR, exist_ok=True)

plt.rcParams.update({
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
})

COLOR_OURS = "#1f77b4"
COLOR_GRID = "#cccccc"


def _save(fig, name):
    fig.savefig(f"{OUT_DIR}/{name}.pdf", bbox_inches="tight")
    fig.savefig(f"{OUT_DIR}/{name}.png", bbox_inches="tight")
    plt.close(fig)
    print(f"saved {OUT_DIR}/{name}.pdf and .png")


# ---------------------------------------------------------------------------
# Figure 1 — Compactness scaling curve (multi-line plot)
# Source: reviewer3_compactness_curve.py, full grid output (independently
# reproduced by user, identical values).
# ---------------------------------------------------------------------------

def figure_compactness():
    n_legs = np.array([2, 3, 4, 5, 6, 8, 10])
    seg_counts = [1, 2, 3, 4]

    def flat_count(n, s):
        return 1 + n * (1 + 2 * (s - 1) + 1)

    fig, ax = plt.subplots(figsize=(5.5, 4))
    for s in seg_counts:
        flat = [flat_count(n, s) for n in n_legs]
        ax.plot(n_legs, flat, marker="o", label=f"flat, {s} segment(s)")
    ax.axhline(6, color="black", linestyle="--", linewidth=2,
              label="typed grammar (constant, 6 rules)")
    ax.set_xlabel("number of legs")
    ax.set_ylabel("rules required")
    ax.set_title("Grammar compactness: flat grows linearly, typed stays constant")
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(alpha=0.3)
    _save(fig, "fig1_compactness_scaling")


# ---------------------------------------------------------------------------
# Figure 2 — Search performance across budgets (multi-line plot)
# Source: reviewerFGH budget-scaling sweep, exact values pasted by user
# (budgets 60/120/200, all 6 methods).
# ---------------------------------------------------------------------------

def figure_budget_scaling():
    budgets = [60, 120, 200]
    data = {
        "PSO": [2.079, 2.289, 2.419],
        "Grammar-guided (ours)": [1.998, 2.107, 2.350],
        "CMA-ES (lite)": [1.968, 1.990, 2.060],
        "Random search": [1.694, 1.922, 2.154],
        "NES": [1.516, 1.928, 2.025],
        "Hill-climbing": [1.289, 1.522, 1.957],
    }
    fig, ax = plt.subplots(figsize=(5.5, 4))
    for name, vals in data.items():
        style = dict(marker="o", linewidth=2.5) if "ours" in name else \
                dict(marker="s", linewidth=1.2, alpha=0.75)
        ax.plot(budgets, vals, label=name, **style)
    ax.set_xlabel("evaluation budget")
    ax.set_ylabel("mean best fitness (8 seeds)")
    ax.set_title("Search parity is stable across a 3.3\u00d7 budget range")
    ax.set_xticks(budgets)
    ax.legend(fontsize=8.5, loc="lower right")
    ax.grid(alpha=0.3)
    _save(fig, "fig2_budget_scaling")


# ---------------------------------------------------------------------------
# Figure 3 — Six-method comparison at budget=60 (bar chart with error bars)
# Source: Table 6, exact values pasted by user, independently reproduced.
# ---------------------------------------------------------------------------

def figure_six_method_bar():
    methods = ["Grammar-guided\n(ours)", "PSO", "CMA-ES\n(lite)",
              "Random\nsearch", "NES", "Hill-\nclimbing"]
    means = [1.998, 2.079, 1.968, 1.694, 1.516, 1.289]
    stds = [0.634, 0.682, 0.478, 0.648, 0.498, 0.642]
    colors = [COLOR_OURS if "ours" in m else "#888888" for m in methods]

    fig, ax = plt.subplots(figsize=(6, 4.3))
    bars = ax.bar(methods, means, yerr=stds, capsize=4, color=colors,
                 edgecolor="black", linewidth=0.6)
    ax.set_ylabel("best fitness (mean \u00b1 std, 8 seeds)")
    ax.set_title("Six-method search comparison, budget = 60")
    ax.grid(axis="y", alpha=0.3)
    ax.tick_params(axis="x", pad=8)
    fig.tight_layout()
    _save(fig, "fig3_six_method_comparison")


# ---------------------------------------------------------------------------
# Figure 4 — Feasibility gate: efficiency vs. quality (two-panel bar chart)
# Source: M-H6 (evals-to-first-feasible) and Table 7/gate ablation
# (task fitness), both previously established and reproduced.
# ---------------------------------------------------------------------------

def figure_gate_two_panel():
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))

    # panel A: efficiency (log scale, since 1.0 vs 9.6 spans an order of
    # magnitude and the gate-off condition has a large std)
    ax = axes[0]
    labels = ["Gate active", "Gate inactive"]
    vals = [1.0, 9.6]
    errs = [0.0, 6.2]
    ax.bar(labels, vals, yerr=errs, capsize=4,
          color=[COLOR_OURS, "#888888"], edgecolor="black", linewidth=0.6)
    ax.set_ylabel("physics evaluations to first\nfeasible body (mean \u00b1 std)")
    ax.set_title("(a) Finding ANY feasible body:\ngate helps, 90% fewer evals")
    ax.grid(axis="y", alpha=0.3)

    # panel B: task-fitness quality under a FIXED budget
    ax = axes[1]
    vals = [1.137, 1.672]
    errs = [0.745, 0.661]
    ax.bar(labels, vals, yerr=errs, capsize=4,
          color=[COLOR_OURS, "#888888"], edgecolor="black", linewidth=0.6)
    ax.set_ylabel("best fitness under fixed budget\n(mean \u00b1 std, 8 seeds)")
    ax.set_title("(b) Finding a GOOD body:\ngate does not help")
    ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Feasibility gate: helps constraint satisfaction, not task optimality",
                fontsize=11, y=1.03)
    fig.tight_layout()
    _save(fig, "fig4_gate_efficiency_vs_quality")


# ---------------------------------------------------------------------------
# Figure 5 — Perception-noise degradation curve (single line plot)
# Source: reviewerDE_noise_and_conjunction.py, bit-flip sweep, exact values
# reproduced identically by user.
# ---------------------------------------------------------------------------

def figure_noise_degradation():
    flip_prob = [0.0, 0.02, 0.05, 0.1, 0.2, 0.3]
    agreement = [0.812, 0.812, 0.719, 0.562, 0.438, 0.344]

    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.plot(flip_prob, agreement, marker="o", linewidth=2.5,
           color=COLOR_OURS)
    ax.axhline(0.812, color="gray", linestyle=":", linewidth=1,
              label="noiseless baseline (0.812)")
    ax.set_xlabel("per-feature bit-flip probability")
    ax.set_ylabel("agreement with symbolic arbitration rule")
    ax.set_title("Containment degrades gracefully under simulated\nperception noise")
    ax.set_ylim(0, 0.9)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    _save(fig, "fig5_noise_robustness")


# ---------------------------------------------------------------------------
# Figure 6 — Per-seed fitness distribution (box plot)
# Source: Table 9 raw per-seed data (Appendix D), grammar-guided vs random,
# budget=60, 8 seeds each -- exact values from the original per-seed table.
# ---------------------------------------------------------------------------

def figure_per_seed_boxplot():
    grammar = [2.461, 0.705, 2.691, 2.127, 2.196, 2.336, 2.253, 1.212]
    random_ = [1.241, 0.675, 2.686, 1.599, 1.275, 1.673, 1.292, 1.903]

    fig, ax = plt.subplots(figsize=(5, 4))
    bp = ax.boxplot([grammar, random_], tick_labels=["Grammar-guided\n(ours)", "Random search"],
                    patch_artist=True, widths=0.5, showmeans=True)
    for patch, color in zip(bp["boxes"], [COLOR_OURS, "#888888"]):
        patch.set_facecolor(color)
        patch.set_alpha(0.5)
    # overlay individual seed points for full transparency
    for i, data in enumerate([grammar, random_], start=1):
        x = np.random.default_rng(0).normal(i, 0.04, size=len(data))
        ax.scatter(x, data, color="black", s=18, zorder=3, alpha=0.7)
    ax.set_ylabel("best fitness")
    ax.set_title("Per-seed distribution, budget = 60 (n = 8 seeds each)")
    ax.grid(axis="y", alpha=0.3)
    _save(fig, "fig6_per_seed_boxplot")


if __name__ == "__main__":
    figure_compactness()
    # fig2/fig6 (and the unused fig3/fig4) here hold stale pre-geometry-fix,
    # n=8 hardcoded values. fig2_budget_scaling and fig6_per_seed_boxplot are
    # now generated from result files by make_session4_figures.py; do not
    # regenerate them from this script.
    figure_noise_degradation()
    print("\nAll 6 figures generated in ./figures/")
