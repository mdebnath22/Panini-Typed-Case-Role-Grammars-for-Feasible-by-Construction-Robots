"""
A convincing head-to-head: grammar-guided search vs. a realistic direct-XML
-edit search, under the SAME evaluation budget, on the SAME task fitness
=============================================================================

Where this fits among this project's search comparisons (read before citing
any one of them as "the" result -- they test different things):

  - m_search_loop.benchmark() / expanded_baseline_table.py: grammar-guided
    search vs. OTHER METHODS THAT ALSO SEARCH OVER THE SAME LEGAL GENOME
    SPACE (random search, CMA-ES, PSO, ...). Everyone there is always
    valid by construction, so this isolates pure OPTIMIZATION POWER --
    and it's a near-tie (Friedman p=0.030, but only coordinate descent
    individually distinguishable after Nemenyi). Honest conclusion there:
    same optimization power as other methods operating in the same legal
    space.
  - complex_environment_search.py: grammar-guided LOCAL MUTATION vs. blind
    random resampling, as the feasible region shrinks. Also not a clean
    win -- traced to a real exploitation/exploration pathology in the
    (mu+lambda) seeding rule, not a grammar advantage.
  - expanded_validity_comparison.py: grammar mutation vs. 14 DIRECT-XML-EDIT
    alternatives, on VALIDITY alone (% of mutations that stay physically
    loadable). Grammar wins outright (100% vs 0-84%) -- but that script
    never asks what this buys you downstream, in an actual task-fitness
    search.

THIS script closes that last gap: it runs a real (mu+lambda) search loop
TWICE, once mutating grammar genomes (always valid) and once mutating raw
MJCF XML directly with the SAME naive edit operators already measured for
validity in expanded_validity_comparison.py (uniform-resample, delete,
duplicate, subtree-swap crossover -- a realistic mix, not the maximally
adversarial or maximally tame extremes in that file's full sweep). BOTH
loops pay the exact same cost per evaluation (1 budget unit, whether the
result is valid or not) and use the exact same CPG task-fitness proxy
(gmm.task_fitness's rollout logic, adapted here to take a compiled XML
string directly rather than requiring a grammar-derived `tree`).

Hypothesis, stated before running: grammar-guided search should win this
one cleanly, because every one of its evaluations spends its budget on a
VALID candidate, while a meaningful fraction of the raw-XML search's
budget gets burned on edits that don't even load/step (fitness -1e3, no
information gained, no new candidate worth keeping) -- turning the
already-measured validity gap into an actual, lower best-fitness-found
under the same fixed budget. If this DOESN'T show up, or the raw-XML
search still keeps pace because its surviving-valid fraction alone is
enough to make progress, report that plainly rather than picking a
sub-metric that looks better.

Run:
    python3 -m experiments.grammar_vs_raw_xml_search --n_seeds 8 --budget 40
"""

from __future__ import annotations

import random
import xml.etree.ElementTree as ET
from typing import Dict, List, Tuple

import numpy as np

from experiments.m_search_loop import (Genome, crossover as grammar_crossover,
                                       evaluate_genome, mutate as grammar_mutate,
                                       random_genome)
from experiments.mh1_xml_baseline import _tree_of
from experiments.expanded_validity_comparison import (
    mutate_delete_element, mutate_duplicate_element, mutate_subtree_swap,
    mutate_uniform_resample)
from gmm.mjcf_compiler import MJCFCompiler
from gmm.task_fitness import _cpg_controls


def _count_actuators(xml: str) -> int:
    try:
        root = ET.fromstring(xml)
    except ET.ParseError:
        return 0
    act = root.find(".//actuator")
    return 0 if act is None else len(list(act))


def evaluate_fitness_from_xml(xml: str, *, horizon_steps: int = 1000,
                              dt_control: int = 4, min_torso_height: float = 0.03,
                              energy_penalty: float = 0.01, seed: int = 0,
                              freq: float = 1.5, amp: float = 0.8) -> float:
    """Same rollout logic as gmm.task_fitness.evaluate_fitness, but takes an
    ALREADY-COMPILED xml string directly (actuator count re-derived from the
    XML itself, not from a live MJCFCompiler instance) -- needed because raw-
    XML mutation changes actuator count (duplicate/delete) in ways a
    grammar-derived `tree` object doesn't go through. Returns a scalar
    fitness (-1e3 on any failure), matching evaluate_genome's convention so
    the two search loops are directly comparable."""
    n_act = _count_actuators(xml)
    if n_act == 0:
        return -1e3
    try:
        import mujoco
    except Exception:
        return -1e3
    try:
        model = mujoco.MjModel.from_xml_string(xml)
        data = mujoco.MjData(model)
        rng = np.random.default_rng(seed)
        phases = (np.linspace(0, 2 * np.pi, n_act, endpoint=False)
                 + rng.normal(0, 0.05, n_act))
        x0 = float(data.qpos[0]) if model.nq > 0 else 0.0
        energies: List[float] = []
        ever_fell = False
        t = 0.0
        for step in range(horizon_steps):
            if step % dt_control == 0:
                ctrl = _cpg_controls(t, n_act, phases, freq=freq, amp=amp)
                data.ctrl[:n_act] = np.clip(ctrl, -1, 1)
                energies.append(float(np.mean(np.abs(ctrl))))
            mujoco.mj_step(model, data)
            t += model.opt.timestep
            if not (np.all(np.isfinite(data.qpos)) and np.all(np.isfinite(data.qvel))):
                ever_fell = True
                break
            torso_z = float(data.qpos[2]) if model.nq > 2 else 1.0
            if torso_z < min_torso_height:
                ever_fell = True
        x1 = float(data.qpos[0]) if model.nq > 0 else 0.0
        displacement = x1 - x0
        if abs(displacement) > 3.0:
            return -1e3
        energy = float(np.mean(energies)) if energies else 0.0
        fall_penalty = 0.3 if ever_fell else 0.0
        return displacement - energy_penalty * energy - fall_penalty
    except Exception:
        return -1e3


def _multi_seed_fitness(xml: str, n_gait_seeds: int = 3) -> float:
    fits = [evaluate_fitness_from_xml(xml, seed=s) for s in range(n_gait_seeds)]
    fits = [f for f in fits if f > -1e3]
    return float(np.mean(fits)) if fits else -1e3


def raw_xml_mutate(xml_a: str, xml_b: str, rng_py: random.Random) -> str:
    """A realistic mix of direct-edit operators (same functions already
    scored for validity in expanded_validity_comparison.py), representing
    how a RoboMorph/LASeR-style direct-XML/parameter edit would actually
    mutate a candidate -- not the single most- or least-adversarial
    extreme, but a believable blend of numeric tuning and structural
    edits a real such method would need to make."""
    roll = rng_py.random()
    if roll < 0.40:
        return mutate_uniform_resample(xml_a, rng_py)
    elif roll < 0.60:
        return mutate_delete_element(xml_a, rng_py)
    elif roll < 0.80:
        return mutate_duplicate_element(xml_a, rng_py)
    else:
        return mutate_subtree_swap(xml_a, xml_b, rng_py)


def grammar_search(budget: int, seed: int, pop_size: int = 8) -> Dict:
    rng = np.random.default_rng(seed)
    population: List[Tuple[float, Genome]] = []
    per_eval_valid: List[bool] = []
    evals_used = 0
    while len(population) < pop_size and evals_used < budget:
        g = random_genome(rng)
        fit, _ = evaluate_genome(g, use_gate=False)
        evals_used += 1
        population.append((fit, g))
        per_eval_valid.append(fit > -1e3)
    while evals_used < budget:
        population.sort(key=lambda x: -x[0])
        survivors = [g for _, g in population[:max(2, pop_size // 2)]]
        if rng.random() < 0.5 and len(survivors) >= 2:
            i, j = rng.choice(len(survivors), 2, replace=False)
            child = grammar_crossover(survivors[i], survivors[j], rng)
        else:
            parent = survivors[rng.integers(len(survivors))]
            child = grammar_mutate(parent, rng)
        fit, _ = evaluate_genome(child, use_gate=False)
        evals_used += 1
        per_eval_valid.append(fit > -1e3)
        population.append((fit, child))
        population.sort(key=lambda x: -x[0])
        population = population[:pop_size]
    best_fit = max(f for f, _ in population)
    valid_frac = float(np.mean(per_eval_valid))  # fraction of EVALS valid, not running-best
    return {"best_fitness": best_fit, "valid_frac": valid_frac}


def raw_xml_search(budget: int, seed: int, pop_size: int = 8) -> Dict:
    rng_np = np.random.default_rng(seed)
    rng_py = random.Random(seed)
    population: List[Tuple[float, str]] = []
    per_eval_valid: List[bool] = []
    evals_used = 0
    while len(population) < pop_size and evals_used < budget:
        g = random_genome(rng_np)
        xml = MJCFCompiler().compile(_tree_of(g))
        fit = _multi_seed_fitness(xml)
        evals_used += 1
        population.append((fit, xml))
        per_eval_valid.append(fit > -1e3)
    while evals_used < budget:
        population.sort(key=lambda x: -x[0])
        survivors = [x for _, x in population[:max(2, pop_size // 2)]]
        a = survivors[rng_py.randrange(len(survivors))]
        b = survivors[rng_py.randrange(len(survivors))]
        child_xml = raw_xml_mutate(a, b, rng_py)
        fit = _multi_seed_fitness(child_xml)
        evals_used += 1
        per_eval_valid.append(fit > -1e3)
        population.append((fit, child_xml))
        population.sort(key=lambda x: -x[0])
        population = population[:pop_size]
    best_fit = max(f for f, _ in population)
    valid_frac = float(np.mean(per_eval_valid))  # fraction of EVALS valid, not running-best
    return {"best_fitness": best_fit, "valid_frac": valid_frac}


def run(n_seeds: int = 8, budget: int = 40, verbose: bool = True) -> Dict:
    g_fits, x_fits, g_valid, x_valid = [], [], [], []
    for s in range(n_seeds):
        g = grammar_search(budget, seed=s)
        x = raw_xml_search(budget, seed=s + 1000)
        g_fits.append(g["best_fitness"]); x_fits.append(x["best_fitness"])
        g_valid.append(g["valid_frac"]); x_valid.append(x["valid_frac"])
        if verbose:
            print(f"  seed={s}: grammar={g['best_fitness']:+8.3f} (valid {100*g['valid_frac']:5.1f}%)   "
                 f"raw-xml={x['best_fitness']:+8.3f} (valid {100*x['valid_frac']:5.1f}%)")

    g_arr, x_arr = np.array(g_fits), np.array(x_fits)
    wins = int((g_arr > x_arr).sum())
    try:
        from scipy import stats
        t_stat, p_val = stats.ttest_rel(g_arr, x_arr)
    except Exception:
        t_stat, p_val = float("nan"), float("nan")

    if verbose:
        print(f"\ngrammar-guided: {g_arr.mean():+.3f} +/- {g_arr.std():.3f}   "
             f"(valid-eval rate {100*np.mean(g_valid):.1f}%)")
        print(f"raw-xml edit  : {x_arr.mean():+.3f} +/- {x_arr.std():.3f}   "
             f"(valid-eval rate {100*np.mean(x_valid):.1f}%)")
        print(f"grammar wins {wins}/{n_seeds} seeds   paired t-test: t={t_stat:.3f}  p={p_val:.4f}")
        sig = p_val == p_val and p_val < 0.05
        print(f"significant at p<0.05: {'YES' if sig else 'NOT YET (more seeds/budget needed)'}")
    return {"g_fits": g_fits, "x_fits": x_fits, "wins": wins, "n_seeds": n_seeds,
           "t_stat": float(t_stat), "p_val": float(p_val),
           "g_valid": g_valid, "x_valid": x_valid}


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_seeds", type=int, default=8)
    ap.add_argument("--budget", type=int, default=40)
    args = ap.parse_args()
    run(n_seeds=args.n_seeds, budget=args.budget)
