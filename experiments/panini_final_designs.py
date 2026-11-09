"""
Re-score PANINI_SEARCH's (and a few baselines') FINAL designs with real RL
training, instead of the cheap CPG proxy
========================================================================

Context / why this script exists (separate from experiments/search21_final_designs.py):
Every RL-rescoring manifest produced earlier used
`experiments.m_search_loop.evolutionary_search` ("grammar_guided_evolution
(ours)", GGE) as "ours" -- see search21_final_designs.py's `OURS_NAME` and
`DIRECT_METHODS`. The paper's actual proposed method is `panini_search`
("PĀṇini"), defined in experiments/m_search_loop_improved.py, NOT
evolutionary_search. This means every RL-rescoring number computed so far
is for the WRONG method. This script re-does the final-design extraction
and MJCF compilation specifically for `panini_search`, alongside GGE (for
an apples-to-apples "did the improved search actually help" comparison)
and a few of the already-established strong baselines, so the eventual RL
re-scoring has something non-trivial to rank PĀṇini against.

This is intentionally a NEW, additive script: it does not import, modify,
or monkeypatch anything in search21_final_designs.py, and it does not touch
m_search_loop_improved.py, m_search_loop.py, main_baseline_table.py, or
expanded_baseline_table.py. It reuses the exact same genome-recovery /
compilation / manifest-writing approach as search21_final_designs.py
(independently reimplemented here against the SAME source modules) because
that approach is already validated and this must be a drop-in-compatible
input to experiments/rl_train_mjx.py and experiments/merge_rl_shards.py.

METHODS INCLUDED (6 total, each x n_seeds):
  - "panini_search (PANINI, proposed)"   -- experiments.m_search_loop_improved.panini_search
                                             *** THE METHOD THIS SCRIPT EXISTS FOR ***
  - "evolutionary_search (GGE)"          -- experiments.m_search_loop.evolutionary_search
                                             (the OLD "ours", kept for a direct comparison)
  - "ipop_cma_es"                        -- experiments.expanded_baseline_table.ALL_METHODS
  - "cma_es_lite"                        -- experiments.expanded_baseline_table.ALL_METHODS
                                             (== main_baseline_table.run_cmaes_lite)
  - "pso"                                -- experiments.expanded_baseline_table.ALL_METHODS
                                             (== main_baseline_table.run_pso)
  - "cross_entropy_method"               -- experiments.expanded_baseline_table.ALL_METHODS

Of these, `panini_search` and `evolutionary_search` already return their
best genome directly as `{"best_fitness", "best_genome", ...}` (see
DIRECT_METHODS below). The other 4 are looked up from ALL_METHODS and
return only a float best fitness, exactly like 19 of the 21 original
methods in search21_final_designs.py -- so the same _GenomeTracker
monkeypatch trick (patch the `evaluate_genome` module-global that
main_baseline_table.py / expanded_baseline_table.py each imported, so
every (genome, fitness) pair any of the 4 baseline methods tries gets
recorded, then pick the recorded genome whose fitness matches the
returned best_fit) is used to recover their best genome too, without
touching a single line of those methods.

SEEDING SCHEME: identical to search21_final_designs.py / this project's
other search-comparison harnesses -- `fn(budget, seed)` /
`fn(budget, seed=seed)` for seed in range(n_seeds), seed = raw seed index
0..n_seeds-1, no offset.

COMPILATION PATH: identical to search21_final_designs.py -- genome ->
BodyGrammar.derive() -> PhiBody().build_tree() -> MJCFCompiler().compile().
This already includes gmm/phi_body.py's 2026-10-23 mounting fix (this
script does not touch that file).

MANIFEST SCHEMA: identical field names/shape to
results_refit/search21_final_designs/manifest.json, so
experiments/rl_train_mjx.py, experiments/merge_rl_shards.py, and
experiments/search21_rl_rescoring_analysis.py can all be pointed at this
manifest's --out_dir with NO code changes (the rescoring-analysis script's
`OURS_NAME` constant is hardcoded to the old GGE string though -- see this
script's printed note at the end; it does not need to match for the
Friedman test / per-method table to run, only for that script's extra
"ours vs rank X" highlighting lines).

Run:
    python3 -m experiments.panini_final_designs \\
        --out_dir results_refit/panini_final_designs \\
        --n_seeds 24 --budget 60
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from typing import Dict, List, Tuple

import numpy as np

from experiments import main_baseline_table as mbt
from experiments import expanded_baseline_table as ebt
from experiments.expanded_baseline_table import ALL_METHODS

# *** THE METHOD THIS SCRIPT EXISTS FOR *** -- imported from
# m_search_loop_improved, NOT m_search_loop. Do not change this import.
from experiments.m_search_loop_improved import panini_search

# GGE ("the old ours"), kept only as a comparison point -- NOT relabeled
# as "ours" here, to avoid repeating the exact mix-up that triggered this
# script's existence.
from experiments.m_search_loop import (
    Genome,
    evaluate_genome as _real_evaluate_genome,
    evolutionary_search,
)
from ptg.body_grammar import BodyGrammar
from ptg.body_sutra import seed_body_rules
from gmm.phi_body import PhiBody
from gmm.mjcf_compiler import MJCFCompiler

PANINI_NAME = "panini_search (PANINI, proposed)"
GGE_NAME = "evolutionary_search (GGE)"

# Methods whose call signature already returns {"best_fitness", "best_genome"}
# directly -- no _GenomeTracker needed.
DIRECT_METHODS: Dict[str, "callable"] = {
    PANINI_NAME: lambda budget, seed: panini_search(budget, seed=seed, verbose=False),
    GGE_NAME: lambda budget, seed: evolutionary_search(budget, seed=seed, verbose=False),
}

# A handful of the already-established strong baselines (all from
# ALL_METHODS, which merges main_baseline_table's BASE_METHODS and
# expanded_baseline_table's NEW_METHODS) -- these return only a float and
# need _GenomeTracker to recover their best genome, exactly like 19 of the
# original 21 methods in search21_final_designs.py.
TRACKED_BASELINE_METHODS = ["ipop_cma_es", "cma_es_lite", "pso", "cross_entropy_method"]
for _name in TRACKED_BASELINE_METHODS:
    assert _name in ALL_METHODS, f"{_name!r} not found in expanded_baseline_table.ALL_METHODS"

ALL_RUN_METHODS = [PANINI_NAME, GGE_NAME] + TRACKED_BASELINE_METHODS


def sanitize(name: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "_", name)
    return re.sub(r"_+", "_", s).strip("_")


class _GenomeTracker:
    """Identical mechanism to search21_final_designs.py's _GenomeTracker
    (reimplemented here rather than imported, to keep this script fully
    independent of that file): monkeypatches the `evaluate_genome` name as
    looked up by main_baseline_table.py and expanded_baseline_table.py so
    every genome any of TRACKED_BASELINE_METHODS tries is recorded."""

    def __init__(self):
        self.record: List[Tuple[Genome, float]] = []

    def __enter__(self):
        def wrapped(genome, **kwargs):
            fit, info = _real_evaluate_genome(genome, **kwargs)
            self.record.append((genome, fit))
            return fit, info

        self._orig_mbt = mbt.evaluate_genome
        self._orig_ebt = ebt.evaluate_genome
        mbt.evaluate_genome = wrapped
        ebt.evaluate_genome = wrapped
        return self

    def __exit__(self, *exc):
        mbt.evaluate_genome = self._orig_mbt
        ebt.evaluate_genome = self._orig_ebt
        return False

    def best_genome_for(self, best_fit: float) -> Genome:
        if not self.record:
            raise RuntimeError("no genomes were recorded -- tracker not active "
                               "during the search call?")
        best, best_diff = None, None
        for genome, fit in self.record:
            diff = abs(fit - best_fit)
            if best_diff is None or diff < best_diff:
                best, best_diff = genome, diff
        if best_diff > 1e-4:
            raise RuntimeError(f"no recorded genome's fitness is close to the "
                               f"method's returned best_fit={best_fit!r} "
                               f"(closest diff={best_diff!r}); reproduction is "
                               f"likely broken, not just noisy.")
        return best


def genome_to_tree(genome: Genome, grammar_seed: int = 0):
    g = BodyGrammar(seed_body_rules(), seed=grammar_seed)
    deriv = g.derive(genome.to_plan())
    return PhiBody().build_tree(deriv)


def run_one(method: str, seed: int, budget: int) -> Dict:
    """Re-run `method` at `seed`, returning its best fitness AND best
    genome. Same call convention as search21_final_designs.py: fn(budget,
    seed) / fn(budget, seed=seed), raw seed index, no offset."""
    if method in DIRECT_METHODS:
        res = DIRECT_METHODS[method](budget, seed)
        return {"fitness": res["best_fitness"], "genome": res["best_genome"]}

    fn = ALL_METHODS[method]
    with _GenomeTracker() as tracker:
        fit = fn(budget, seed)
    genome = tracker.best_genome_for(fit)
    return {"fitness": fit, "genome": genome}


def validate_mjcf(xml_path: str, settle_steps: int = 50) -> Dict:
    """Load + step the compiled MJCF in real MuJoCo (CPU) to confirm it
    actually loads and steps without crashing -- same validation standard
    used everywhere else this session (search21_final_designs.py,
    rank_correlation_harness.py)."""
    report = {"loaded": False, "stepped": False, "error": None, "n_actuators": None}
    try:
        import mujoco
        model = mujoco.MjModel.from_xml_path(xml_path)
        report["loaded"] = True
        report["n_actuators"] = int(model.nu)
        data = mujoco.MjData(model)
        for _ in range(settle_steps):
            mujoco.mj_step(model, data)
        if np.all(np.isfinite(data.qpos)) and np.all(np.isfinite(data.qvel)):
            report["stepped"] = True
    except Exception as e:
        report["error"] = f"{type(e).__name__}: {e}"
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=str,
                    default="results_refit/panini_final_designs")
    ap.add_argument("--n_seeds", type=int, default=24)
    ap.add_argument("--budget", type=int, default=60)
    ap.add_argument("--methods_limit", type=int, default=None,
                    help="For a quick smoke test: only run the first N methods.")
    ap.add_argument("--seeds_limit", type=int, default=None,
                    help="For a quick smoke test: only run the first N seeds.")
    ap.add_argument("--skip_mujoco_validate", action="store_true")
    args = ap.parse_args()

    methods = list(ALL_RUN_METHODS)
    assert PANINI_NAME in methods, "panini_search must be included -- do not remove it"
    if args.methods_limit:
        methods = methods[: args.methods_limit]
    n_seeds = args.seeds_limit or args.n_seeds

    os.makedirs(os.path.join(args.out_dir, "bodies"), exist_ok=True)

    bodies: List[Dict] = []
    run_failures: List[Dict] = []
    t_start = time.time()
    infeasible_count = 0
    infeasible_by_method: Dict[str, int] = {}

    for mi, method in enumerate(methods):
        for seed in range(n_seeds):
            t0 = time.time()
            try:
                out = run_one(method, seed, args.budget)
            except Exception as e:
                print(f"[{method} seed={seed}] FAILED: {e}")
                run_failures.append({"method": method, "seed": seed, "error": str(e)})
                continue
            fit, genome = out["fitness"], out["genome"]

            body_id = f"{sanitize(method)}__seed{seed}"
            xml_name = f"{body_id}.xml"

            geometric_feasible = True
            validate = {}
            tree = None
            try:
                tree = genome_to_tree(genome)
            except Exception as e:
                geometric_feasible = False
                validate = {"error": f"grammar/tree build failed: {e}"}

            xml = None
            if tree is not None:
                try:
                    xml = MJCFCompiler().compile(tree, name=body_id)
                except Exception as e:
                    geometric_feasible = False
                    validate = {"error": f"MJCF compile failed: {e}"}

            xml_path = os.path.join(args.out_dir, "bodies", xml_name)
            if xml is not None:
                with open(xml_path, "w") as f:
                    f.write(xml)
                if not args.skip_mujoco_validate:
                    validate = validate_mjcf(xml_path)
                    if not (validate.get("loaded") and validate.get("stepped")):
                        geometric_feasible = False

            entry = {
                "body_id": body_id,
                "method": method,
                "seed": seed,
                "genome": {
                    "n_legs": genome.n_legs,
                    "segments_per_leg": genome.segments_per_leg,
                    "seg_len": genome.seg_len,
                    "seg_rad": genome.seg_rad,
                    "angle_jitter": genome.angle_jitter,
                },
                "proxy_fitness": fit,
                "mjcf_path": os.path.relpath(xml_path, args.out_dir),
                "geometric_feasible": bool(geometric_feasible),
                "validate": validate,
            }
            bodies.append(entry)

            if not geometric_feasible:
                infeasible_count += 1
                infeasible_by_method[method] = infeasible_by_method.get(method, 0) + 1

            if (mi * n_seeds + seed) % 25 == 0:
                elapsed = time.time() - t_start
                print(f"[{elapsed:7.1f}s] {method!r} seed={seed}: fit={fit:+.4f} "
                     f"feasible={geometric_feasible}")

    manifest = {
        "label": f"panini_final_designs ({len(methods)} methods x {n_seeds} seeds)",
        "n_bodies": len(bodies),
        "n_geometrically_feasible": sum(1 for b in bodies if b["geometric_feasible"]),
        "proxy_controller": "fixed sinusoidal CPG (gmm.task_fitness.evaluate_fitness), "
                            "same cheap proxy used throughout this project's search comparisons",
        "search_budget": args.budget,
        "n_seeds": n_seeds,
        "seeding_scheme": "seed = raw seed index 0..n_seeds-1, passed directly as "
                         "panini_search(budget, seed=seed) / evolutionary_search(budget, seed=seed) "
                         "/ fn(budget, seed); no offset.",
        "methods": methods,
        "panini_method_name": PANINI_NAME,
        "gge_method_name": GGE_NAME,
        "bodies": bodies,
        "gpu_side_contract": {
            "what_to_produce": (
                "For each body_id in this manifest with geometric_feasible=true, train "
                "a per-body locomotion policy on the body's MJCF with "
                "experiments/rl_train_mjx.py (same script/contract used for "
                "search21_final_designs.py's manifests), writing results back via "
                "--body_id/--body_index shards; merge with experiments/merge_rl_shards.py."
            ),
            "then_run": (
                "python3 -m experiments.search21_rl_rescoring_analysis "
                f"--out_dir {args.out_dir}  "
                "(works unmodified for the Friedman test / per-method mean-rank table; "
                "that script's OURS_NAME constant is hardcoded to "
                "'grammar_guided_evolution (ours)' i.e. GGE_NAME-ish, not PANINI_NAME, "
                "so its extra 'ours vs rank X' highlighting lines will silently not "
                "fire for PANINI_NAME unless OURS_NAME is edited to "
                f"{PANINI_NAME!r} in that file -- the main per-method comparison table "
                "and statistical test do not depend on OURS_NAME and work either way.)"
            ),
        },
    }
    manifest_path = os.path.join(args.out_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    summary = {
        "n_methods": len(methods),
        "methods": methods,
        "panini_method_name": PANINI_NAME,
        "n_seeds": n_seeds,
        "n_bodies": len(bodies),
        "n_run_failures": len(run_failures),
        "run_failures": run_failures,
        "n_infeasible": infeasible_count,
        "infeasible_by_method": infeasible_by_method,
        "wall_clock_s": time.time() - t_start,
        "manifest": manifest_path,
    }
    summary_path = os.path.join(args.out_dir, "panini_final_designs_build_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 70)
    print(f"Done. {len(bodies)} bodies across {len(methods)} methods x {n_seeds} seeds.")
    print(f"Methods: {methods}")
    print(f"Run failures: {len(run_failures)} / {len(bodies)}")
    if run_failures:
        print("First few run failures:")
        for m in run_failures[:10]:
            print(f"  {m}")
    print(f"Infeasible bodies: {infeasible_count} / {len(bodies)}  by method: {infeasible_by_method}")
    print(f"Manifest: {manifest_path}")
    print(f"Build summary: {summary_path}")


if __name__ == "__main__":
    main()
