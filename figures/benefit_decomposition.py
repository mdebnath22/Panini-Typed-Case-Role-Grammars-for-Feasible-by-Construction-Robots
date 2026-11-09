"""
Benefit decomposition: optimization power vs. validity/compactness
====================================================================

Answers one question directly, in numbers: "if the grammar ties with
CMA-ES on raw task fitness, what is it actually buying us?"

All numbers below are taken VERBATIM from prior runs already reported
in the paper (expanded_baseline_table / Friedman+Nemenyi output,
expanded_validity_comparison's 15-condition table, reviewer3's
compactness sweep) -- nothing here is re-derived or estimated. Each
block below cites which script/run it came from so a reader can verify
against the raw output.

Produces fig7_benefit_decomposition.{png,pdf}: a 2-panel figure.
  Left panel  -- search/optimization axis: Friedman average rank (lower
                 is better) for grammar-guided search vs. the CMA-ES
                 family and PSO. These are statistically tied (see the
                 p-values annotated on the bars) -- NO advantage here.
  Right panel -- validity axis: percent of mutations that remain legal,
                 grammar-guided vs. every direct-XML-edit alternative
                 that was actually run. This is where the gap is.

Run:
    python3 figures/benefit_decomposition.py
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Block 1 -- search/optimization axis
# Source: expanded_baseline_table.py's 21-method run (Friedman + Nemenyi).
# Friedman chi^2 = 33.46, p = 0.030 (an omnibus effect exists across all 21).
# Nemenyi post-hoc: ONLY coordinate descent is individually distinguishable
# from grammar-guided search (p = 0.0095); every other pairwise comparison,
# including the full CMA-ES family, did not survive correction -- but the
# exact average-rank VALUE for each of the other 19 methods (CMA-ES
# standard/lite, IPOP-CMA-ES, PSO, etc.) was not re-extracted from a saved
# per-method results file for this figure, and must not be guessed. Only
# grammar-guided (6.25, the best of 21) and PSO (6.50, the only other
# value actually quoted in prior output) are plotted as real bars below.
# Everything else is reported as text, not fabricated bar heights.
#
# CMA-ES lite vs. standard (reviewerFGH_cmaes_budget_controller.py, part
# G): a SEPARATE paired t-test, not a Friedman rank -- p = 0.816.
# ---------------------------------------------------------------------------
RANK_METHODS = ["grammar-guided\n(ours)", "PSO", "coordinate\ndescent"]
RANK_VALUES = [6.25, 6.50, None]  # avg Friedman rank, lower=better; coord.
                                  # descent's own rank value isn't on hand
                                  # either -- only its p-value vs ours is.
RANK_SIGNIFICANT = [False, False, True]
RANK_NOTE = "p=0.0095 vs ours\n(Nemenyi; rank value\nnot separately on hand)"
RANK_CAVEAT = ("Only grammar-guided (6.25, best of 21) and PSO (6.50) have a\n"
              "confirmed average-rank value on hand from prior output. The other\n"
              "18 methods, including the full CMA-ES family, are tied with ours\n"
              "per Nemenyi (p>0.05) but their exact rank values aren't re-plotted\n"
              "here without re-reading expanded_baseline_table.py's saved output.\n"
              "CMA-ES lite vs. standard: separate paired t-test p=0.816.")

# ---------------------------------------------------------------------------
# Block 2 -- validity axis
# Source: expanded_validity_comparison.py's 15-condition table (grammar +
# pure_reinit + 9 single-XML mutators + 2 two-XML mutators + 2 extra-k),
# each out of 100 trials, as already reported in the paper's validity table.
# ---------------------------------------------------------------------------
VALIDITY_METHODS = ["grammar\n(ours)", "pure_reinit", "mild_only",
                    "range_\nconstrained", "adversarial", "uniform",
                    "k=5", "extreme", "k=10", "delete", "subtree\nswap",
                    "crossover", "duplicate", "bitflip", "latent"]
VALIDITY_PCT = [100, 100, 100, 100, 84, 78, 72, 62, 50, 39, 16, 8, 0, 0, 0]

# ---------------------------------------------------------------------------
# Block 3 -- compactness axis (annotated as a third fact, not a third panel)
# Source: reviewer3_compactness_curve.py -- 6 typed rules regardless of body
# size vs. 17/37/49 flat rules for 4x2/6x3/8x3 bodies (2.8x/6.2x/8.2x), up to
# 13.5x at the largest body in the sweep (10 legs x 4 segments).
# ---------------------------------------------------------------------------
COMPACTNESS_NOTE = ("Compactness (reviewer3_compactness_curve.py): 6 typed rules\n"
                    "at every body size vs. 17/37/49 flat rules (4x2/6x3/8x3 legs x\n"
                    "segments) -- 2.8x/6.2x/8.2x, up to 13.5x in the full sweep.")


def make_figure(outpath_png: str = "fig7_benefit_decomposition.png",
                outpath_pdf: str = "fig7_benefit_decomposition.pdf") -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 6.0))

    # Left: search/optimization parity. Only two bars are real confirmed
    # values (ours, PSO); coordinate descent's rank value isn't on hand, so
    # it is annotated as text rather than plotted as a fabricated bar.
    ax = axes[0]
    plotted_methods = [m for m, v in zip(RANK_METHODS, RANK_VALUES) if v is not None]
    plotted_values = [v for v in RANK_VALUES if v is not None]
    colors = ["#2f855a"] + ["#2b6cb0"] * (len(plotted_values) - 1)
    bars = ax.bar(plotted_methods, plotted_values, color=colors, width=0.5)
    ax.set_ylabel("average Friedman rank (lower = better)")
    ax.set_title("(a) Search/optimization axis: statistical tie")
    ax.axhline(plotted_values[0], color="#2f855a", linestyle="--", linewidth=1, alpha=0.6)
    for bar in bars[1:]:
        ax.annotate("n.s.", xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                   xytext=(0, 4), textcoords="offset points",
                   ha="center", fontsize=8, color="#2b6cb0")
    ax.set_ylim(0, max(plotted_values) * 1.6)
    ax.text(0.5, 0.92, RANK_NOTE, transform=ax.transAxes, ha="center", va="top",
            fontsize=7.5, color="#c53030")
    plt.setp(ax.get_xticklabels(), fontsize=8)

    # Right: validity gap
    ax2 = axes[1]
    colors2 = ["#2f855a"] + ["#c53030" if v < 100 else "#2b6cb0" for v in VALIDITY_PCT[1:]]
    ax2.bar(VALIDITY_METHODS, VALIDITY_PCT, color=colors2)
    ax2.set_ylabel("mutation validity rate (%)")
    ax2.set_title("(b) Validity axis: grammar-guided is always 100%")
    ax2.set_ylim(0, 108)
    ax2.axhline(100, color="#2f855a", linestyle="--", linewidth=1, alpha=0.6)
    plt.setp(ax2.get_xticklabels(), rotation=55, ha="right", fontsize=7)

    fig.suptitle("Where the grammar's advantage actually is: not raw search power, "
                 "but validity and compactness", fontsize=10.5, y=0.99)
    fig.tight_layout(rect=[0, 0.22, 1, 0.95])
    fig.text(0.5, 0.01, RANK_CAVEAT + "\n\n" + COMPACTNESS_NOTE,
            ha="center", va="bottom", fontsize=7.3, color="#444")
    fig.savefig(outpath_png, dpi=200, bbox_inches="tight")
    fig.savefig(outpath_pdf, bbox_inches="tight")
    print(f"wrote {outpath_png} and {outpath_pdf}")


def print_summary_table() -> None:
    print("\nSummary -- same optimization power, different validity/compactness\n")
    print(f"{'method':22s} {'avg rank':>9s} {'tied w/ ours?':>14s}")
    for m, r, sig in zip(RANK_METHODS, RANK_VALUES, RANK_SIGNIFICANT):
        tied = "no (p<0.01)" if sig else "yes (n.s.)"
        r_str = f"{r:9.2f}" if r is not None else "   (n/a)"
        print(f"{m.replace(chr(10),' '):22s} {r_str} {tied:>14s}")
    print()
    print(f"{'mutation method':22s} {'validity %':>11s}")
    for m, v in zip(VALIDITY_METHODS, VALIDITY_PCT):
        print(f"{m.replace(chr(10),' '):22s} {v:11d}")
    print()
    print(COMPACTNESS_NOTE)


if __name__ == "__main__":
    print_summary_table()
    make_figure()
