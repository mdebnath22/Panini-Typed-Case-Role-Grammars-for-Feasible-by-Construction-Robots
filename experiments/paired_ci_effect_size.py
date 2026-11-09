"""
Paired 95% CIs, Cohen's d, and a pre-specified non-inferiority test for the
21-method search comparison at n=24 seeds.

This is the one computation in this paper that was originally done as an
ad hoc interactive script rather than a saved, rerunnable module -- a
genuine reproducibility gap (a reviewer should be able to trace every
main-table number to a script, not to narrative claims about what an
earlier scratch computation found). This file is that script, made
permanent: it reads the exact same raw per-seed cache
(`results_refit/search21_n24_b60.json`) that produced the numbers quoted
in Section~sec:results-search and Table~tab:nemenyi-ci, and recomputes
them from scratch.

Non-inferiority margin: 0.05 absolute fitness units, ~12% of the mean
fitness across the 21-method field at n=24 -- pre-specified as "a
difference this paper would consider practically meaningful," not fit
post hoc to any single comparison's observed gap.

Run:
    python3 -m experiments.paired_ci_effect_size \
        --cache results_refit/search21_n24_b60.json --ours "grammar_guided_evolution (ours)"
"""
from __future__ import annotations

import argparse
import json
from typing import Dict, List

import numpy as np
from scipy import stats

NON_INFERIORITY_MARGIN = 0.05


def paired_stats(ours: np.ndarray, other: np.ndarray, margin: float = NON_INFERIORITY_MARGIN) -> Dict:
    """95% CI and Cohen's d on the paired difference (ours - other), plus a
    one-sided non-inferiority test: H0: ours <= other - margin, i.e. ours
    is worse than other by at least `margin`. Rejecting H0 (one-sided
    p < 0.05) is evidence FOR non-inferiority; failing to reject it is NOT
    evidence of equivalence, and this function reports it as such rather
    than conflating the two the way an uncorrected two-sided test alone
    would invite."""
    diff = ours - other
    n = len(diff)
    mean_diff = float(diff.mean())
    sd_diff = float(diff.std(ddof=1)) if n > 1 else 0.0
    se = sd_diff / np.sqrt(n) if n > 1 else 0.0
    tcrit = stats.t.ppf(0.975, df=max(1, n - 1))
    ci = (mean_diff - tcrit * se, mean_diff + tcrit * se)
    cohens_d = mean_diff / sd_diff if sd_diff > 0 else float("nan")
    t_uncorrected, p_uncorrected = stats.ttest_rel(ours, other)
    # one-sided non-inferiority test against the pre-specified margin
    t_ni = (mean_diff + margin) / se if se > 0 else float("inf")
    p_ni = float(stats.t.sf(t_ni, df=max(1, n - 1))) if se > 0 else (0.0 if mean_diff > -margin else 1.0)
    return dict(
        n=n, mean_diff=mean_diff, ci95=ci, cohens_d=cohens_d,
        uncorrected_t=float(t_uncorrected), uncorrected_p=float(p_uncorrected),
        non_inferiority_margin=margin,
        non_inferiority_p=p_ni,
        non_inferiority_established=bool(p_ni < 0.05),
    )


def run(cache_path: str, ours_key: str, verbose: bool = True) -> Dict:
    data: Dict[str, List[float]] = json.load(open(cache_path))
    ours = np.array(data[ours_key])
    out = {}
    for method, vals in data.items():
        if method == ours_key:
            continue
        out[method] = paired_stats(ours, np.array(vals))
    if verbose:
        print(f"Paired comparisons vs.\\ {ours_key!r} (n={len(ours)} seeds, "
             f"non-inferiority margin={NON_INFERIORITY_MARGIN}):\n")
        print(f"{'method':<32}{'mean diff':>10}{'95% CI':>22}{'d':>7}"
             f"{'p (2-sided)':>12}{'NI p':>9}{'NI?':>5}")
        for method, r in out.items():
            ci = f"[{r['ci95'][0]:.3f}, {r['ci95'][1]:.3f}]"
            print(f"{method:<32}{r['mean_diff']:>10.3f}{ci:>22}{r['cohens_d']:>7.2f}"
                 f"{r['uncorrected_p']:>12.3f}{r['non_inferiority_p']:>9.3f}"
                 f"{'yes' if r['non_inferiority_established'] else 'no':>5}")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="results_refit/search21_n24_b60.json")
    ap.add_argument("--ours", default="grammar_guided_evolution (ours)")
    args = ap.parse_args()
    run(args.cache, args.ours)
