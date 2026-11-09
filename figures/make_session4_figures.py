"""
Session-4 figures for the paper rewrite -- every value is READ FROM A RESULT
FILE at run time (no hardcoded numbers), except where a figure explicitly
plots a Sol-terminal-reported summary statistic, which is labelled as such
in the figure itself and in its LaTeX caption.

Regenerates (overwriting the stale, pre-geometry-fix versions):
  fig2_budget_scaling      n=24 budget sweep, original grammar-guided evolution
  fig6_per_seed_boxplot    n=24 per-seed distribution, PĀṇini vs tied cluster
  fig8_proxy_vs_rl         proxy-vs-RL rank correlation, stale vs rebuilt catalog
New:
  fig9_search30_ranks      (defined but not called: dropped for compactness; the
                           condensed main-text table carries the same content)
  fig10_rl_rescoring       proxy rank vs RL rank of each method's final designs
  fig11_investigations     forest plot over the competitive-edge investigations

Run:  python3 make_session4_figures.py   (writes .pdf/.png here and copies
the PDFs into ../paper/)
"""
from __future__ import annotations

import json
import os
import re
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from scipy.stats import studentized_range

S = "/tmp/claude-0/-home-claude/c18217aa-55f9-5a60-af83-dd7ac3a3e402/scratchpad"
ROOT = os.path.join(S, "build/typed-case-role-grammars")
HERE = os.path.dirname(os.path.abspath(__file__))
PAPER = os.path.join(ROOT, "paper")

# --- palette: validated with dataviz/scripts/validate_palette.js (light) ----
C_PANINI = "#2a78d6"      # categorical slot 1 (blue)  -- PĀṇini family
C_GGE = "#eb6834"         # categorical slot 2 (orange) -- original (mu+lambda)
C_BASE = "#8f8e8a"        # neutral marks for every generic baseline
C_TEXT = "#0b0b0b"
C_TEXT2 = "#52514e"
C_GRID = "#e4e3df"

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": C_TEXT2, "axes.labelcolor": C_TEXT, "xtick.color": C_TEXT2,
    "ytick.color": C_TEXT2, "text.color": C_TEXT, "axes.linewidth": 0.8,
    "lines.linewidth": 2.0, "font.family": "DejaVu Sans", "pdf.fonttype": 42,
    "figure.dpi": 150, "savefig.dpi": 200,
})

PRETTY = {
    "PĀṇini (new)": "PĀṇini", "panini_search (plain)": "PĀṇini",
    "panini_search (PANINI)": "PĀṇini",
    "grammar_guided_evolution (GGE)": "Grammar-guided evol. (GGE)",
    "grammar_guided_evolution (ours)": "Grammar-guided evol.",
    "grammar_guided_evolution_v2 (new)": "Grammar-guided evol. v2",
    "ipop_cma_es": "IPOP-CMA-ES", "cma_es_lite": "CMA-ES (simplified)",
    "cross_entropy_method": "Cross-entropy method", "pso": "PSO",
    "coarse_to_fine": "Coarse-to-fine", "turbo": "TuRBO",
    "lhs_plus_refinement": "LHS + refinement", "random_search": "Random search",
    "hill_climbing": "Hill-climbing", "nes": "NES",
    "augmented_random_search": "Augmented random search",
    "diversity_archive_es": "Diversity-archive ES", "graph_heuristic_search": "Graph heuristic search",
    "shade": "SHADE", "bayesian_optimization": "Bayesian optimization",
    "novelty_search": "Novelty search", "spsa": "SPSA", "differential_evolution": "Differential evolution",
    "llm_proposer": "LLM proposer", "one_plus_one_es": "(1+1)-ES", "map_elites": "MAP-Elites",
    "simulated_annealing": "Simulated annealing", "latent_space_bo": "Latent-space BO",
    "coordinate_descent": "Coordinate descent", "finetuned_controller": "Fine-tuned controller",
    "archive_criterion_rotation": "Archive criterion-rotation", "nelder_mead": "Nelder-Mead",
}


def L(p):
    with open(os.path.join(S, p)) as f:
        return json.load(f)


def _save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(HERE, f"{name}.{ext}"), bbox_inches="tight")
    shutil.copy(os.path.join(HERE, f"{name}.pdf"), os.path.join(PAPER, f"{name}.pdf"))
    plt.close(fig)
    print("saved", name)


def _grid(ax, axis="y"):
    ax.grid(axis=axis, color=C_GRID, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)


def mean_ranks(d):
    names = list(d)
    M = np.array([d[k] for k in names]).T
    R = np.apply_along_axis(lambda r: stats.rankdata(-r), 1, M).mean(0)
    return dict(zip(names, R)), M.shape[0]


def paired_ci(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = a - b
    n = len(d)
    h = stats.t.ppf(0.975, n - 1) * d.std(ddof=1) / np.sqrt(n)
    return d.mean(), d.mean() - h, d.mean() + h, stats.ttest_rel(a, b).pvalue


# ---------------------------------------------------------------------------
def fig2_budget_scaling():
    """Source: scratchpad/panini_budget_sweep_n24.json (n=24, budgets
    60/120/200, 8 methods, PĀṇini -- the CEM-based searcher with the
    std-floor fix -- against GGE (the original (mu+lambda) method) and six
    generic optimizers)."""
    bs = L("panini_budget_sweep_n24.json")
    budgets = [60, 120, 200]
    methods = list(bs["60"]["raw"].keys())
    fig, ax = plt.subplots(figsize=(5.6, 3.5))
    P = "panini_search (PANINI)"
    end = {}
    for m in methods:
        mu = np.array([np.mean(bs[str(b)]["raw"][m]) for b in budgets])
        se = np.array([np.std(bs[str(b)]["raw"][m], ddof=1) / np.sqrt(len(bs[str(b)]["raw"][m])) for b in budgets])
        if m == P:
            ax.fill_between(budgets, mu - 1.96 * se, mu + 1.96 * se, color=C_PANINI, alpha=0.15, lw=0)
            ax.plot(budgets, mu, color=C_PANINI, marker="o", ms=5, lw=2.2, zorder=4)
        else:
            col = C_GGE if "GGE" in m else C_BASE
            ax.plot(budgets, mu, color=col, marker="o", ms=3.5, lw=1.1, alpha=0.9, zorder=3)
        end[m] = mu[-1]
    # direct labels at the right edge, nudged apart to avoid collisions
    order = sorted(end, key=end.get)
    ys, last = [], -1
    for m in order:
        y = max(end[m], last + 0.0105)
        ys.append((m, y)); last = y
    for m, y in ys:
        ax.annotate(PRETTY.get(m, m), xy=(200, end[m]), xytext=(206, y), fontsize=7.5,
                    color=C_PANINI if m == P else (C_GGE if "GGE" in m else C_TEXT2), va="center",
                    fontweight="bold" if m == P else "normal",
                    arrowprops=dict(arrowstyle="-", color=C_GRID, lw=0.6))
    ax.set_xticks(budgets)
    ax.set_xlim(52, 262)
    ax.set_xlabel("evaluation budget (fitness queries)")
    ax.set_ylabel("mean best proxy fitness ($n=24$ seeds)")
    ax.set_title("Budget sweep: Pāṇini (band: 95% CI) vs. GGE and six generic optimizers",
                 loc="left", color=C_TEXT)
    _grid(ax)
    _save(fig, "fig2_budget_scaling")


# ---------------------------------------------------------------------------
TIED = ["ipop_cma_es", "cross_entropy_method", "pso", "coarse_to_fine", "turbo",
        "lhs_plus_refinement", "cma_es_lite"]


def fig6_per_seed():
    """Source: scratchpad/work/full30_n24.json (n=24, budget 60, post-fix)."""
    d = L("work/full30_n24.json")
    keys = ["PĀṇini (new)", "grammar_guided_evolution (ours)"] + TIED
    keys = sorted(keys, key=lambda k: -np.mean(d[k]))
    fig, ax = plt.subplots(figsize=(6.4, 3.3))
    rng = np.random.default_rng(0)
    for i, k in enumerate(keys):
        v = np.asarray(d[k])
        col = C_PANINI if k.startswith("PĀṇini") else (C_GGE if k.startswith("grammar_guided_evolution (") else C_BASE)
        bp = ax.boxplot([v], positions=[i], widths=0.55, patch_artist=True, showfliers=False,
                        medianprops=dict(color=C_TEXT, lw=1.2), whiskerprops=dict(color=C_TEXT2, lw=0.8),
                        capprops=dict(color=C_TEXT2, lw=0.8))
        bp["boxes"][0].set(facecolor=col, alpha=0.22, edgecolor=col, linewidth=1.0)
        ax.scatter(i + rng.uniform(-0.17, 0.17, len(v)), v, s=9, color=col, alpha=0.85, zorder=3,
                   edgecolor="white", linewidth=0.4)
        ax.scatter([i], [v.mean()], marker="D", s=26, color="white", edgecolor=C_TEXT, zorder=4, linewidth=0.9)
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels([PRETTY[k] for k in keys], rotation=30, ha="right")
    ax.set_ylabel("best proxy fitness, budget 60")
    ax.set_title("Per-seed best fitness, $n=24$ seeds (dots: seeds; diamond: mean)", loc="left")
    _grid(ax)
    _save(fig, "fig6_per_seed_boxplot")


# ---------------------------------------------------------------------------
def _fisher_ci(r, n, kind):
    # Fieller-Hartley-Pearson approximate SEs on the Fisher-z scale
    se = (1.06 / np.sqrt(n - 3)) if kind == "spearman" else np.sqrt(0.437 / (n - 4))
    z = np.arctanh(r)
    return np.tanh(z - 1.96 * se), np.tanh(z + 1.96 * se)


def fig8_proxy_vs_rl():
    """Stale catalog: scratchpad/rank_correlation_result.json (n=28, local file).
    Rebuilt catalog: Sol terminal output (n=31) transcribed in
    paper_rewrite_inputs/rank_corr_v2_terminal_output.txt -- parsed here, not
    retyped. No per-body RL values for the rebuilt catalog exist locally, so
    no scatter can be drawn for it; we plot the coefficients with
    approximate 95% CIs instead."""
    old = L("rank_correlation_result.json")
    txt = open(os.path.join(S, "paper_rewrite_inputs/rank_corr_v2_terminal_output.txt")).read()
    n_new = int(re.search(r"n = (\d+) matched", txt).group(1))
    rho_new, prho_new = map(float, re.search(r"Spearman rho = ([\d.\-]+)\s+\(p = ([\d.]+)\)", txt).groups())
    tau_new, ptau_new = map(float, re.search(r"Kendall tau\s+= ([\d.\-]+)\s+\(p = ([\d.]+)\)", txt).groups())
    rows = [
        ("Spearman $\\rho$", "spearman", old["spearman_rho"], old["spearman_p"], rho_new, prho_new),
        ("Kendall $\\tau$", "kendall", old["kendall_tau"], old["kendall_p"], tau_new, ptau_new),
    ]
    fig, ax = plt.subplots(figsize=(5.6, 2.3))
    y = 0
    yt, yl = [], []
    for lab, kind, r_o, p_o, r_n, p_n in rows:
        for (r, p, n, col, mk, tag, dy) in [
            (r_o, p_o, old["n"], C_BASE, "o", f"pre-fix catalog, $n={old['n']}$ (local file)", 0.18),
            (r_n, p_n, n_new, C_PANINI, "s", f"rebuilt catalog, $n={n_new}$ (Sol-reported)", -0.18)]:
            lo, hi = _fisher_ci(r, n, kind)
            ax.plot([lo, hi], [y + dy] * 2, color=col, lw=2.0, solid_capstyle="round")
            ax.scatter([r], [y + dy], marker=mk, s=34, color=col, zorder=3, edgecolor="white", linewidth=0.6)
            ax.text(hi + 0.03, y + dy, f"{r:+.3f}  ($p={p:.3f}$)", va="center", fontsize=7.5, color=C_TEXT2)
        yt.append(y); yl.append(lab); y -= 1
    ax.axvline(0, color=C_TEXT2, lw=0.8, ls=(0, (3, 3)))
    ax.set_yticks(yt); ax.set_yticklabels(yl)
    ax.set_xlim(-0.6, 1.05)
    ax.set_ylim(y + 0.45, 0.55)
    ax.set_xlabel("rank correlation, CPG-proxy fitness vs. RL-trained fitness (bars: approx. 95% CI)")
    h1 = plt.Line2D([], [], color=C_BASE, marker="o", lw=2, label=rows[0][1] and f"pre-fix catalog, $n={old['n']}$ (local file)")
    h2 = plt.Line2D([], [], color=C_PANINI, marker="s", lw=2, label=f"rebuilt post-fix catalog, $n={n_new}$ (Sol-reported)")
    ax.legend(handles=[h1, h2], loc="lower center", frameon=False, fontsize=7.5, ncol=2, bbox_to_anchor=(0.5, 1.0))
    _grid(ax, "x")
    _save(fig, "fig8_proxy_vs_rl")


# ---------------------------------------------------------------------------
def fig9_search30():
    """Source: scratchpad/work/full30_n24.json. Nemenyi CD at alpha=0.05."""
    d = L("work/full30_n24.json")
    R, n = mean_ranks(d)
    k = len(R)
    cd = studentized_range.ppf(0.95, k, np.inf) / np.sqrt(2) * np.sqrt(k * (k + 1) / (6.0 * n))
    order = sorted(R, key=R.get)
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    rp = R["PĀṇini (new)"]
    ax.axvspan(rp - cd, rp + cd, color=C_PANINI, alpha=0.08, lw=0, zorder=0)
    for i, m in enumerate(order):
        col = C_PANINI if m == "PĀṇini (new)" else (C_GGE if m == "grammar_guided_evolution (ours)" else C_BASE)
        mk = "D" if col != C_BASE else "o"
        ax.plot([rp, R[m]], [i, i], color=C_GRID, lw=0.8, zorder=1)
        ax.scatter([R[m]], [i], s=30 if col != C_BASE else 18, color=col, marker=mk, zorder=3,
                   edgecolor="white", linewidth=0.5)
    ax.set_yticks(range(k))
    ax.set_yticklabels([PRETTY.get(m, m) for m in order], fontsize=7)
    for t, m in zip(ax.get_yticklabels(), order):
        if m == "PĀṇini (new)":
            t.set_color(C_PANINI); t.set_fontweight("bold")
        elif m == "grammar_guided_evolution (ours)":
            t.set_color(C_GGE); t.set_fontweight("bold")
    ax.invert_yaxis()
    ax.set_xlabel("mean rank across 24 seeds (lower is better; 30 methods)")
    ax.text(rp + cd + 0.4, 2.0, f"band edge = PĀṇini $\\pm$ Nemenyi\ncritical difference ({cd:.2f} ranks,\n$\\alpha=0.05$, $k=30$, $n=24$)",
            fontsize=7.5, color=C_PANINI, va="center")
    ax.set_title("30-method comparison, budget 60: everything inside the band is\n"
                 "statistically indistinguishable from PĀṇini", loc="left")
    _grid(ax, "x")
    _save(fig, "fig9_search30_ranks")
    return cd


# ---------------------------------------------------------------------------
def fig10_rl_rescoring():
    """RL mean rank per method: paper_rewrite_inputs/rl_rescoring_full672_28methods.json
    (rl_mean_rank only -- the file's proxy_mean_rank_matched_subset is NaN and
    is NOT used). Proxy mean rank: computed here from the per-body
    rerun_proxy_fitness of the SAME 672 bodies in
    results_refit/search21_final_designs/manifest.json."""
    rl = L("paper_rewrite_inputs/rl_rescoring_full672_28methods.json")
    man = json.load(open(os.path.join(ROOT, "results_refit/search21_final_designs/manifest.json")))["bodies"]
    meths = rl["methods"]
    M = np.zeros((24, len(meths)))
    for b in man:
        M[b["seed"], meths.index(b["method"])] = b["rerun_proxy_fitness"]
    pr = dict(zip(meths, np.apply_along_axis(lambda r: stats.rankdata(-r), 1, M).mean(0)))
    rr = rl["rl_mean_rank"]
    rho, p = stats.spearmanr([pr[m] for m in meths], [rr[m] for m in meths])
    fig, ax = plt.subplots(figsize=(4.9, 4.3))
    lim = (3, 25)
    ax.plot(lim, lim, color=C_GRID, lw=1.0, zorder=0)
    ax.text(23.6, 24.4, "equal rank", fontsize=7, color=C_TEXT2, ha="right", rotation=0)
    for m in meths:
        col = C_GGE if m.startswith("grammar_guided") else C_BASE
        ax.scatter(pr[m], rr[m], s=34 if col != C_BASE else 20, color=col, zorder=3,
                   marker="D" if col != C_BASE else "o", edgecolor="white", linewidth=0.5)
    lab = {"coordinate_descent": (-1.0, 2.6, "Coordinate descent\n(one identical design, all 24 seeds)"),
           "grammar_guided_evolution (ours)": (2.2, 3.3, "Grammar-guided evol."),
           "ipop_cma_es": (0.7, -1.4, "IPOP-CMA-ES"),
           "nelder_mead": (-6.5, 0.8, "Nelder-Mead"),
           "finetuned_controller": (0.6, -3.4, "Fine-tuned\ncontroller"),
           "map_elites": (-1.5, 1.6, "MAP-Elites")}
    for m, (dx, dy, t) in lab.items():
        ax.annotate(t, xy=(pr[m], rr[m]), xytext=(pr[m] + dx, rr[m] + dy), fontsize=7,
                    color=C_GGE if m.startswith("grammar") else C_TEXT2,
                    arrowprops=dict(arrowstyle="-", color=C_TEXT2, lw=0.5))
    ax.set_xlim(*lim); ax.set_ylim(*lim)
    ax.invert_xaxis(); ax.invert_yaxis()
    ax.set_xlabel("proxy (CPG) mean rank of final designs")
    ax.set_ylabel("RL-trained mean rank of the same designs")
    ax.set_title(f"28 methods, 24 seeds each: method-level\nSpearman $\\rho={rho:.2f}$ ($p={p:.2f}$); better is up/right",
                 loc="left")
    _grid(ax, "both")
    _save(fig, "fig10_rl_rescoring")
    return rho, p


def fig10b_panini_rl_rescoring():
    """PĀṇini-specific RL re-scoring: rl_rescoring_result_panini.json
    (144 bodies, 6 methods incl. panini_search, 24 seeds each, all converged).
    Not significant (Friedman p=0.71), so plotted without a trend claim."""
    d = L("rl_rescoring_result_panini.json")
    meths = d["methods"]
    pr = d["proxy_mean_rank_matched_subset"]
    rr = d["rl_mean_rank"]
    fig, ax = plt.subplots(figsize=(4.8, 4.3))
    lim = (2.6, 5.3)
    ax.plot(lim, lim, color=C_GRID, lw=1.0, zorder=0)
    for m in meths:
        is_panini = m.startswith("panini_search")
        is_gge = m.startswith("evolutionary_search")
        col = C_PANINI if is_panini else (C_GGE if is_gge else C_BASE)
        ax.scatter(pr[m], rr[m], s=50 if (is_panini or is_gge) else 32, color=col, zorder=3,
                   marker="D" if (is_panini or is_gge) else "o", edgecolor="white", linewidth=0.6)
    # points cluster tightly (rank range ~3-5 of 6), so use leader-line labels
    # spread around the cluster rather than inline offsets.
    lab = {
        "panini_search (PANINI, proposed)": (-0.55, 0.55, "Pāṇini"),
        "evolutionary_search (GGE)": (0.55, 0.55, "GGE"),
        "ipop_cma_es": (-0.75, -0.25, "IPOP-CMA-ES"),
        "cma_es_lite": (0.65, 0.05, "CMA-ES (simplified)"),
        "pso": (0.55, -0.55, "PSO"),
        "cross_entropy_method": (-0.35, -0.65, "Cross-entropy"),
    }
    for m, (dx, dy, t) in lab.items():
        is_panini = m.startswith("panini_search")
        is_gge = m.startswith("evolutionary_search")
        ax.annotate(t, xy=(pr[m], rr[m]), xytext=(pr[m] + dx, rr[m] + dy), fontsize=7.5,
                    color=C_PANINI if is_panini else (C_GGE if is_gge else C_TEXT2),
                    fontweight="bold" if (is_panini or is_gge) else "normal",
                    arrowprops=dict(arrowstyle="-", color=C_GRID, lw=0.6))
    ax.set_xlim(*lim); ax.set_ylim(*lim)
    ax.invert_xaxis(); ax.invert_yaxis()
    ax.set_xlabel("proxy (CPG) mean rank of final designs")
    ax.set_ylabel("RL-trained mean rank of the same designs")
    ax.set_title(f"6 methods, 24 seeds each (144 bodies, all converged):\nFriedman $\\chi^2={d['chi2']:.2f}$, $p={d['p_friedman']:.2f}$ -- not significant",
                 loc="left")
    _grid(ax, "both")
    _save(fig, "fig10b_panini_rl_rescoring")


# ---------------------------------------------------------------------------
def _parse_log_seeds(path, methods):
    out = {m: [] for m in methods}
    for line in open(os.path.join(S, path)):
        if re.search(r"seed=\d+:", line):
            for m in methods:
                mm = re.search(re.escape(m) + r"=([+\-][\d.]+)", line)
                if mm:
                    out[m].append(float(mm.group(1)))
    return out


def investigation_rows():
    """Each row: (label, variant per-seed, best generic baseline per-seed,
    baseline name, plain-PĀṇini per-seed or None, n). 'Best generic baseline'
    = the non-PĀṇini method with the highest mean in that same experiment
    (chosen post hoc, which is the conservative choice for the variant)."""
    rows = []

    def best_generic(res, exclude):
        cands = {k: v for k, v in res.items() if k not in exclude}
        k = max(cands, key=lambda k: np.mean(cands[k]))
        return k, cands[k]

    h = L("screening_result.json")["results"]
    bk, bv = best_generic(h, ["panini_search (hetero)"])
    rows.append(("#4 Heterogeneous per-limb space (b=80)", h["panini_search (hetero)"], bv, bk, None))
    nz = _parse_log_seeds("noise_confirm_030.log", ["panini_search", "ipop_cma_es", "cross_entropy_method", "pso"])
    bk, bv = best_generic(nz, ["panini_search"])
    rows.append(("#5 Fitness noise, 30% (b=60)", nz["panini_search"], bv, bk, None))
    tr = _parse_log_seeds("transfer_confirm.log", ["panini_search", "ipop_cma_es", "cross_entropy_method", "pso"])
    bk, bv = best_generic(tr, ["panini_search"])
    rows.append(("#6 Straight-line objective (b=60)", tr["panini_search"], bv, bk, None))
    for b in ("60", "120"):
        g = L(f"panini_gp_logs/confirm_b{b}.json")["results"]
        bk, bv = best_generic(g, ["panini_search_gp (ours)", "panini_search (hetero, plain)"])
        rows.append((f"#7 Subtree crossover, hetero (b={b})", g["panini_search_gp (ours)"], bv, bk,
                     g["panini_search (hetero, plain)"]))
    dp = L("dpp_out/compare_n24.json")
    dres = dp["results"] if "results" in dp else dp
    dk = [k for k in dres if "DPP" in k][0]
    bk, bv = best_generic(dres, [dk, "panini_search (plain)"])
    rows.append(("#8 DPP elite selection (b=60)", dres[dk], bv, bk, dres["panini_search (plain)"]))
    mo = L("compare_result.json")["results"]
    bk, bv = best_generic(mo, ["panini_search_motif", "panini_search"])
    rows.append(("#9 Motif reuse (b=60)", mo["panini_search_motif"], bv, bk, mo["panini_search"]))
    te = L("tempering_compare_n24_b60.json")["results"]
    bk, bv = best_generic(te, ["panini_search_tempering", "panini_search"])
    rows.append(("#10 Parallel tempering (b=60)", te["panini_search_tempering"], bv, bk, te["panini_search"]))
    for tag, f in [("b=60", "curric_logs/compare_n24_b60.json"), ("b=120", "curric_logs/compare_n24_b120.json"),
                   ("b=60", "curric_n48_b60.json"), ("b=120", "curric_n48_b120.json")]:
        c = L(f)["results"]
        bk, bv = best_generic(c, ["panini_search_curriculum (new)", "panini_search (plain)"])
        rows.append((f"#11 Curriculum over depth ({tag})", c["panini_search_curriculum (new)"], bv, bk,
                     c["panini_search (plain)"]))
    t = L("panini_transfer_out/confirm24.json")["results"]
    tk = "panini_transfer (hetero, held-out)"
    bk, bv = best_generic(t, [tk, "panini_cold (hetero, held-out)"])
    rows.append(("#12 Held-out composition (b=80)", t[tk], bv, bk, t["panini_cold (hetero, held-out)"]))
    ca = L("panini_ca_logs/default_spec_n24.json")["fitness"]
    ck = "panini_search_constraint_aware"
    bk, bv = best_generic(ca, [ck, "panini_search (slot-acc.)", "panini_search_gate", "hand_coded_sampler"])
    rows.append(("#13 Constraint-aware repair (b=60)", ca[ck], bv, bk, ca["panini_search (slot-acc.)"]))
    nc = L("niche_logs/confirm_budget35_n24.json")["results"]
    bk, bv = best_generic(nc, ["PAaNini (new)", "grammar_guided_evolution (ours)"])
    rows.append(("#14 Extreme-low budget (b=35)", nc["PAaNini (new)"], bv, bk, None))
    return rows


def fig11_investigations():
    rows = investigation_rows()
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 4.9), sharey=True, gridspec_kw=dict(width_ratios=[1.15, 1]))
    table = []
    for i, (lab, var, base, bk, plain) in enumerate(rows):
        n = len(var)
        m, lo, hi, p = paired_ci(var, base)
        col = C_PANINI
        filled = lo > 0 or hi < 0
        axes[0].plot([lo, hi], [i, i], color=col, lw=1.8, solid_capstyle="round")
        axes[0].scatter([m], [i], s=28, marker="o", zorder=3, linewidth=1.2,
                        facecolor=col if filled else "white", edgecolor=col)
        axes[0].text(1.02, i, f"{PRETTY.get(bk.split(' (')[0].replace('_cold',''), bk.split(' (')[0])}", transform=axes[0].get_yaxis_transform(),
                     fontsize=6.3, color=C_TEXT2, va="center")
        row = dict(label=lab, n=n, baseline=bk, diff=m, lo=lo, hi=hi, p=p)
        if plain is not None:
            m2, lo2, hi2, p2 = paired_ci(var, plain)
            filled2 = lo2 > 0 or hi2 < 0
            axes[1].plot([lo2, hi2], [i, i], color=C_PANINI, lw=1.8, solid_capstyle="round", alpha=0.85)
            axes[1].scatter([m2], [i], s=28, marker="s", zorder=3, linewidth=1.2,
                            facecolor=C_PANINI if filled2 else "white", edgecolor=C_PANINI)
            row.update(diff_plain=m2, lo_plain=lo2, hi_plain=hi2, p_plain=p2)
        else:
            axes[1].text(0, i, "n/a (plain PĀṇini is the method)", fontsize=6.3, color=C_TEXT2, ha="center", va="center")
        table.append(row)
    for ax in axes:
        ax.axvline(0, color=C_TEXT2, lw=0.8, ls=(0, (3, 3)))
        _grid(ax, "x")
    axes[0].set_yticks(range(len(rows)))
    axes[0].set_yticklabels([r[0] + f"  [$n$={len(r[1])}]" for r in rows], fontsize=7)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("variant $-$ strongest generic baseline\n(paired mean diff., 95% CI; baseline named at right)")
    axes[1].set_xlabel("variant $-$ plain PĀṇini\n(isolates the added mechanism)")
    axes[0].set_title("(a) vs. best generic optimizer in that run", loc="left")
    axes[1].set_title("(b) vs. plain PĀṇini", loc="left")
    fig.text(0.01, -0.02, "Filled marker: uncorrected paired 95% CI excludes 0. Hollow: CI includes 0. "
             "Fitness units differ across substrates (#4, #7, #12 use the heterogeneous genome).",
             fontsize=6.8, color=C_TEXT2)
    fig.tight_layout()
    _save(fig, "fig11_investigations")
    with open(os.path.join(S, "rewrite_work/forest_rows.json"), "w") as f:
        json.dump(table, f, indent=1, default=float)
    for r in table:
        print("  %-42s n=%2d vs %-24s %+.3f [%+.3f,%+.3f] p=%.3f | plain %s" % (
            r["label"], r["n"], r["baseline"][:24], r["diff"], r["lo"], r["hi"], r["p"],
            "" if "diff_plain" not in r else "%+.3f [%+.3f,%+.3f] p=%.3f" % (r["diff_plain"], r["lo_plain"], r["hi_plain"], r["p_plain"])))


if __name__ == "__main__":
    fig2_budget_scaling()
    fig6_per_seed()
    fig8_proxy_vs_rl()
    print("  method-level proxy-vs-RL:", fig10_rl_rescoring())
    fig11_investigations()
