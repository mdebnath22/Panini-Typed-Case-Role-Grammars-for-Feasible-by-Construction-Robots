# Reproducing this paper

Pinned environment: `requirements.txt` (exact versions) or `Dockerfile`
(`docker build -t typed-case-role-grammars . && docker run --rm -it
typed-case-role-grammars bash`). The search statistics in particular are
sensitive to MuJoCo's minor version (Appendix "Reproducibility Notes"),
so results from an unpinned environment should not be expected to match
exactly, even with identical seeds.

Every fitness-dependent result was re-run after the free-joint compiler
fix described in Appendix "Implementation Notes and Pitfalls"
(`gmm/mjcf_compiler.py`); logs for that rerun are under `results_refit/`.
Feasibility-only results that never touch MuJoCo dynamics (the
`reviewer_objections_*` family, the compactness/homomorphism/compgen
results) were unaffected by that bug and were not rerun; their original
logs are under `results/`.

This file maps every numbered table and figure to the script that
produces it. "Command" is runnable from the repository root once
dependencies are installed.

## Method section (Table 1, Figure 1, Table 2 homomorphism, Table 3 compactness, Table 4 compgen)

| Item | Script | Notes |
|---|---|---|
| Table~tab:homomorphism (defect) | `experiments/reviewer2_defect_decomposition.py`, `experiments/reviewer_q1q5_longer_chains.py` | Feasibility-only; not rerun post-fix (doesn't touch MuJoCo dynamics). |
| Table~tab:compactness | `experiments/todo1_typing_vs_inheritance.py` | Pure rule-counting; no simulator involved. |
| Table~tab:compgen | `experiments/todo2_morphology_compgen.py`, `experiments/reviewer_q3_ood_diagnostic.py` | Trains the $\Phi_\theta$ transformer; seed-dependent, reported seed in script defaults. |

## Boxes / containment (Table 5)

| Item | Script | Notes |
|---|---|---|
| Table~tab:boxes rows 1-5 (untrained/trained/regularization/held-out) | `experiments/reviewer_q2_region_sensitivity.py` (`run(seed=0)`) | Single-seed sanity checks on containment structure itself. |
| Table~tab:boxes rows 6-8 (box/order-embedding/Euclidean decision agreement, mean $\pm$ std) | `experiments/multiseed_containment.py` | `python3 -m experiments.multiseed_containment` — reruns `reviewer_q2_region_sensitivity.run` at 10 seeds. Log: `results_refit/multiseed_containment.log`. |
| Figure~fig:noise (perception-noise degradation) | `experiments/reviewerDE_noise_and_conjunction.py` | |

## Main search/validity results

| Item | Script | Log |
|---|---|---|
| Table~tab:search21 (n=8) | `experiments/main_baseline_table.py`, `experiments/expanded_baseline_table.py` | `results_refit/search21_n8_b60.log` |
| Table~tab:search21-scaled (n=24) | same, `--n_seeds 24`, plus `experiments/search21_scaled_stats.py` for the Friedman/Nemenyi summary | `results_refit/search21_n24_b60.log` / `.json` |
| Table~tab:nemenyi-ci (paired CI / Cohen's d / non-inferiority) | `experiments/paired_ci_effect_size.py` | `python3 -m experiments.paired_ci_effect_size --cache results_refit/search21_n24_b60.json`. Log: `results_refit/paired_ci_effect_size.log`. |
| Table~tab:recent-sota (7 LLM/co-design analogues) | `experiments/recent_sota_baselines.py` | `results_refit/recent_sota_n8_b60.log` |
| Table~tab:recent-sota-2 (RoboGrammar GHS, GLSO) | `experiments/robogrammar_glso_baselines.py` | `results_refit/robogrammar_glso.log` |
| Table~tab:gate (2-condition gate ablation) | `experiments/mh6b_gate_ablation.py` | `results_refit/mh6b_gate.log` |
| Table~tab:gait-gate (3-condition, gait-correlated) | `experiments/gait_correlated_gate_ablation.py` | `results_refit/gait_gate.log` |
| Table~tab:validity21 (15-condition validity comparison) | `experiments/expanded_validity_comparison.py` | `results_refit/validity21_main.log` |
| Table~tab:stricter-validity (loads+steps vs. locomotes) | `experiments/stricter_validity_comparison.py` | `results_refit/stricter_validity.log` |
| Table~tab:complex-env, tab:complex-env-fixed | `experiments/complex_environment_search.py` | `results_refit/complex_env.log`, `complex_env_fixed.log` |
| Table~tab:counterfactual (careful vs. naive editors) | `experiments/counterfactual_structural_validity.py` | |
| Table~tab:inheritance-ablation | `experiments/inheritance_ablation_validity.py` | |

## Hard design space / coupled feasibility / "four objections" (Appendix)

| Item | Script | Cache |
|---|---|---|
| Table~tab:validity-hard, tab:new-feasible, tab:topology | `experiments/validity15_hard_environment.py`, `experiments/validity15_displacement.py`, `experiments/topology_change_search.py` | Feasibility-only; unaffected by the compiler bug. |
| Table~tab:coupled-feasibility | `experiments/coupled_feasibility_validity.py` | |
| §sec:coupled-feasibility "Is this the grammar, or just evenly spaced limbs?" paragraph (hand-coded, non-grammar generator vs. grammar vs. hand-coded-random-angle) | `experiments/hand_coded_coordinated_body.py` | `python3 -m experiments.hand_coded_coordinated_body`. Log: `results_refit/hand_coded_coordinated.log`. Builds the identical tree shape with no `BodyGrammar`/`BodySutra`/`ParibhasaResolver` import anywhere in its call path. |
| Table~tab:objections, tab:e1-random-walk, tab:e1-search, tab:e2-separability, tab:e3-editors, tab:e4-factorial | `experiments/reviewer_objections_hard_space.py <E1walk|E1search|E2|E3|E4> 8`, `experiments/reviewer_objections_sensitivity.py` | `results/obj_<which>_8.json`. Feasibility/material-objective only; unaffected by the compiler bug, not rerun. |

## Closed kinematic chains

| Item | Script |
|---|---|
| Appendix "Closed Kinematic Chains" (four-bar linkage, standalone solver + residual check) | `experiments/closed_kinematic_chain.py` |
| §app:closed-chain grammar-integrated `CLOSE_LOOP` rule (repair/reject, downstream MuJoCo `<equality><connect>` simulation) | `experiments/closed_loop_grammar.py --n_trials 300`. Log: `results_refit/closed_loop_grammar.log`. The additive `resolve_fn` hook it depends on lives in `ptg/body_sutra.py`; every pre-existing `BodySutra` leaves it `None`, verified by `pytest tests/ -q` (5 passed, unchanged) before and after. |
| Table~tab:budget-sweep (budgets 60/120/200), "standard vs. lite CMA-ES" comparison (§sec:budget-sweep) | `experiments/reviewerFGH_cmaes_budget_controller.py --which F` (CMA-ES variant), `--which G` (budget sweep). Logs: `results_refit/cmaes_standard_vs_lite.log`, `results_refit/budget_sweep.log`. |
| §app:pitfalls contact-sensitivity study (penetration distribution, stricter-contact rerun, repair/rejection stage) | `experiments/contact_sensitivity.py --n_bodies 200`. Log: `results_refit/contact_sensitivity.log`. Depends on the additive `xml_override` parameter on `gmm/task_fitness.py`'s `evaluate_fitness` (default `None`, every other call site unaffected). |
| Table~tab:manip-benchmark (3-task physical manipulation benchmark, 3 planners: grammar / uninformed baseline / informed-but-unstaged baseline) | `experiments/manipulation_benchmark.py --n_trials 150`. Log: `results_refit/manipulation_benchmark.log`. |
| §sec:effort-reuse extension-cost comparison ("knee" limbs, grammar vs. non-modular hand-coded vs. modular-registry hand-coded) | `experiments/effort_reuse_study.py` (imports `experiments/modular_hand_coded_generator.py` for the third, registry-based comparator). Log: `results_refit/effort_reuse_study.log`. Does not edit `experiments/hand_coded_coordinated_body.py` or any file in `ptg/`/`gmm/`; measures cost mechanically via `inspect.getsource` + `difflib`, not by hand. |

## Re-run under the corrected simulator, with a reversed finding

**Table~tab:xml-vs-grammar-search / Section~sec:validity-vs-fitness**
(live-search validity-vs-fitness comparison) —
`experiments/grammar_vs_raw_xml_search.py --n_seeds 8 --budget 40`. Log:
`results_refit/grammar_vs_raw_xml_search.log`. The pre-fix run reported
a statistical tie ($1.499\pm0.615$ vs. $1.342\pm0.485$, $p=0.571$); the
corrected-simulator rerun instead shows the raw-XML alternative
directionally ahead ($0.934\pm0.658$ vs. $0.379\pm0.092$, grammar wins
$0/8$ seeds, $p=0.069$) — a change in direction, not only magnitude, so
this is reported as a superseded finding rather than a quietly updated
number.

## Does the repair/rejection stage change the main search conclusion?

**§app:pitfalls, "Three further checks" (fourth check)** — Section~sec:budget-sweep
and Table~tab:search21 report grammar-guided search's headline number
($0.422\pm0.141$, budget 60, $n=8$ seeds) with no penetration check at
all. `experiments/repair_stage_search_check.py --n_seeds 8 --budget 60`
wraps `m_search_loop.evaluate_genome` (unmodified) with the same
threshold/repair/reject logic as `contact_sensitivity.py` and re-runs the
identical search. Log: `results_refit/repair_stage_search_check.log`.
Result: $0.4226\pm0.1402$ vs.\ the no-repair $0.4220\pm0.1410$
(one-sample $t=0.011$, $p=0.991$) — unchanged within noise, with
$472/480$ genomes accepted as drawn, $4$ repaired, $4$ rejected
($0.8\%$ rejection rate).

## Explicitly NOT re-run (flagged, not attempted)

1. **Appendix~sec:repro** (simulator-version sensitivity, MuJoCo 3.8.1
   vs. 3.13.0) — this check inherently requires two different installed
   MuJoCo versions and is flagged in the paper text itself as an open
   item rather than re-run here.

The CMA-ES budget-sweep item previously listed here has since been rerun
under the corrected simulator (see Table~tab:budget-sweep above) and is
no longer pending.

## Known, disclosed deviation from exact independent reproduction

The multi-seed containment numbers in `multiseed_containment.py` and the
paired-stats numbers in `paired_ci_effect_size.py` were originally
computed as one-off interactive scripts during revision and have been
rewritten here as permanent, rerunnable modules with the same inputs;
their output matches the numbers quoted in the paper to the reported
precision (verified by diffing `results_refit/*.log` against the
in-text numbers).
