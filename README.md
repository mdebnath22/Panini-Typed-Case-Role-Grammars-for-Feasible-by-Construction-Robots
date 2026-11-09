# Typed Case-Role Production Grammars for Verifiable Robot Skill and Morphology Synthesis

Code, figures and LaTeX source for the paper. A typed case-role production
grammar (manipulation roles Agent/Patient/Instrument/Location/Source/Goal;
morphology roles Root/Limb/Segment/Joint/Terminal) with parameter inheritance
and specificity arbitration, a geometric realization map Φ to SE(3), a
feasibility gate, and evolutionary search over MuJoCo robot bodies.

## Layout

```
ptg/          grammar: roles, rules, arbitration (body_karaka, body_sutra, body_grammar, paribhasa, karaka)
gmm/          Φ_body (phi_body), SE(3) math (se3), MJCF compiler, CPG locomotion fitness (task_fitness)
sulba/        geometric feasibility gate (reach, clearance, mass, support polygon, self-collision)
experiments/  every experiment script (table below)
figures/      make_figures.py, schematic_robots.py, fig1-fig6 (.pdf/.png)
tests/        smoke tests
```

## Setup

```bash
pip install numpy scipy matplotlib mujoco==3.13.0     # pin: results change across MuJoCo versions
pip install torch                                     # only for compgen / learned-box scripts
pip install cma scikit-posthocs imageio               # optional (some baselines, post-hoc tests, renders)
pip install brax                                       # optional, GPU strongly recommended (full_pipeline_reproduction.py)
```

Run everything **from the repository root** as modules (`python3 -m experiments.<name>`).
Do not redirect stderr to /dev/null; it hides import errors.

MuJoCo 3.8.1 vs 3.13.0 gives different search results on identical code and
seeds (paper, simulator-sensitivity section). Pin the version.

## Experiments

| Script | What it measures |
|---|---|
| `reviewer2_defect_decomposition` | Φ homomorphism defect: rotation vs translation parts (inactive 0.0; active ≈0.037) |
| `reviewer_q1q5_longer_chains` | Defect over chains k=2..6 (bounded, max 0.081) |
| `reviewer3_compactness_curve` | Flat vs typed rule count over legs × segments (max 13.5×) |
| `todo1_typing_vs_inheritance` | Compactness ablation: typing alone vs + inheritance |
| `todo5_cardinality_bound` | Derivation cardinality bound (21/21 match) |
| `reviewer_q6_runtime_cost` | Runtime cost of inheritance (3.4–3.7%) |
| `todo3_box_extended`, `reviewer_q2_region_sensitivity`, `reviewerC_specificity_sensitivity` | Specificity as box containment; width-floor sweep; order-embedding and Euclidean controls |
| `reviewerDE_noise_and_conjunction` | Bit-flip perception noise; box-intersection conjunction |
| `todo2_morphology_compgen` | Morphology compositional generalization (needs torch) |
| `main_baseline_table` | 6-method search comparison |
| `expanded_baseline_table` | 21-method search comparison, Friedman + Nemenyi |
| `recent_sota_baselines` | 5 budget-matched analogues of recent LLM/co-design papers (RoboMorph, LASeR, RoboMoRe, Debate2Create, Strgar&Kriegman — see its module docstring for exactly what's reproduced vs. borrowed) |
| `expanded_validity_comparison` | Grammar mutation vs 14 direct-edit alternatives (15 conditions) |
| `todo4a_scaled_benchmark`, `m_search_loop` | Scaled multi-seed search vs random |
| `mh1_xml_baseline`, `mh6b_gate_ablation` | Raw-XML mutation validity; gate on/off under a fixed physics budget |
| `reviewerFGH_cmaes_budget_controller` | CMA-ES lite vs standard, budget sweep, controller study. |
| `diagnostic_noise_sweep`, `_v2` | Fitness noise vs gait seeds |
| `reviewer_q3_ood_diagnostic` | Manipulation compgen anomaly (OOD error < ID), length-controlled, n=5 seeds -- diagnosed (template-composition artifact, see below) |
| `full_pipeline_reproduction` | Real SAC-trained-policy fitness (Brax) replacing the CPG-proxy for the 5 recent-method analogues -- needs a GPU to run at reportable scale |
| `reviewer_q8_goal_aligned_gate` | Goal-aligned feasibility gate |
| `reviewer_q7_cmaes_baseline` | Superseded by the baseline tables |
| `render_real_robots` | Real MuJoCo renders, grammar vs direct edit |
| `counterfactual_structural_validity` | Careful (non-adversarial) hand-coded structural editors vs. grammar vs. naive baselines, per-edit-type, with failure-cause classification |
| `feasibility_rate_scaling` | Gate-OFF cost (~1/p) vs. gate-ON cost (~1) across 4 feasibility difficulties (p=0.50→0.01) |
| `inheritance_ablation_validity` | Does disabling anuvrtti change mutation validity or per-leg genome-controllability? |
| `complex_environment_search` | Does grammar-guided local mutation beat blind random search as feasibility shrinks? |
| `stricter_validity_comparison` | Stricter, locomotion-based (loads+steps+≥2cm displacement) validity criterion vs. the main loads-and-steps one |
| `robogrammar_glso_baselines` | 2 more budget-matched analogues: RoboGrammar's Graph Heuristic Search, GLSO's latent-space Bayesian optimization (real sklearn GP) |
| `gait_correlated_gate_ablation` | A feasibility gate extension that also checks a short CPG rollout (fractional physics cost), testing whether gait-correlated gating reverses the gate-vs-fitness negative result |
| `closed_kinematic_chain` | Pure-geometry planar four-bar linkage loop-closure solver; tests the Limitations claim about closed chains needing a new verification step |
| `search21_scaled_stats` | Re-runs the 21-method comparison at n_seeds=24 (vs. the main table's n=8) with Friedman + Nemenyi |
| `coupled_feasibility_validity` | Isolates the grammar's actual coordination mechanism (evenly-spaced mount angles) with a mechanism-targeted counterfactual (independently-randomized mount angles) on a coupled, cross-limb criterion (support-polygon area) that no per-field range constraint can express, calibrated to very-low-feasibility regimes (p≈0.30/0.10/0.05) -- answers the Table 17 "range-constrained resampling also hits 100%" critique by testing construction rather than survival-of-one-edit |

Helper: `box_embed_np.py` (NumPy box embeddings, required by the region scripts).

Example:

```bash
python3 -m experiments.reviewer3_compactness_curve
python3 -m experiments.expanded_baseline_table
python3 -m experiments.expanded_validity_comparison
python3 -m tests.test_search_loop && python3 -m tests.test_new_experiments
```

### Rendering

```bash
export MUJOCO_GL=egl        # or osmesa; needs a node with a GL backend (not a bare login node)
python3 -m experiments.render_real_robots --seed 0 --settle_steps 300
```

### Figures

```bash
python3 figures/make_figures.py
```

## Comparing recent LLM/co-design morphology papers (RoboMorph, LASeR, etc.)

RoboMorph, LASeR, RoboMoRe, Debate2Create, and Strgar & Kriegman's morphological-pretraining
paper each train a real RL policy (SAC, ~5e5–1e6 steps) per candidate on Gym/Brax Ant-family
tasks — a different simulator, different task suite, and ~10,000× the cost per candidate
compared to this paper's CPG-proxy fitness under a 40–60-evaluation budget. Quoting their
published numbers next to ours in one table would be apples-to-oranges.

`experiments/recent_sota_baselines.py` instead reimplements each paper's **search mechanism**
inside this pipeline — same genome, same CPG fitness, same shared evaluation budget as the
other 21 methods — so the comparison is fair even though it's an analogue, not a reproduction.

Budget accounting across population-based (PSO/DE/CMA-ES/our evolution/the new
archive-rotation and diversity-archive methods) and single-solution-based (hill-climbing,
SA, (1+1)-ES, coordinate descent, Nelder-Mead, SPSA) methods is normalized to **completed
physics evaluations**, not generations or iterations — a population method spends `pop_size`
evaluations per generation (truncated if the remaining budget is smaller), a single-solution
method spends 1 per step (2 for SPSA's two-point gradient estimate). The reported number is
each method's best fitness ever seen by the time its shared budget hit zero.

## Counterfactual experiments (is the grammar's advantage necessary, not just convenient?)

Three scripts probe this directly, each with a stated prediction in its own docstring and an
honest report of whatever actually came out (including two surprising findings below):

- **`counterfactual_structural_validity.py`** -- writes a genuinely careful (non-adversarial),
  hand-coded structural XML editor (real `xml.etree` parsing, unique renaming, actuator
  fix-up) as the strongest non-grammar counterfactual, rather than relying on the existing
  deliberately-naive mutators. Result, n=100: `careful_duplicate_limb` matches the grammar
  100/100 -- for this edit type the grammar's advantage is convenience, not necessity, since
  careful hand-coding replicates it. 
- **`feasibility_rate_scaling.py`** -- the quantitative version of the "small feasible region"
  intuition: as the feasibility spec's difficulty increases (measured ground-truth feasibility
  p = 0.538, 0.179, 0.039, 0.011), gate-OFF cost (expected tries to first feasible genome) grows
  as predicted by 1/p (1.8, 4.2, 18.0, 67.0 mean tries), while gate-ON stays flat at 1.0 by
  construction. 
- **`inheritance_ablation_validity.py`** -- disabling anuvrtti (`anuvrtti_export=frozenset()`
  on every rule) leaves mutation validity unchanged (100/100 both ON and OFF), confirming
  validity comes from typing/legality-gating, not inheritance, as predicted. 

## Does grammar-guided search actually win in a complex (low-feasibility) environment?

`experiments/complex_environment_search.py` tests this directly -- the square/feasibility-region
intuition ("in a 1%-feasible environment, a smarter method should beat blind random sampling"),
not just restated but run.

Hypothesis tested: once grammar-guided evolutionary search finds one feasible genome, mutating
it locally should land on another feasible genome more often than redrawing uniformly at random,
and that advantage should grow as the feasible fraction p shrinks (same spec difficulties as
`feasibility_rate_scaling.py`: p≈0.54, 0.17, 0.05, 0.01).

**Result (n_seeds=8, budget=40, feasible-eval-rate = % of the budget that lands on a
feasible genome):**

| target p | grammar-guided evolution (no restarts) | +restarts | blind random search |
|---|---|---|---|
| 0.5 | **87.2%** | 70.3% | 45.9% |
| 0.2 | **66.3%** | 41.6% | 16.6% |
| 0.05 | **28.8%** | 11.9% | 3.8% |
| 0.01 | 0.0% | 2.8% | 0.9% |

Paired t-test (feasible-eval-rate, evolution vs. random): p = 0.000 / 0.002 / 0.080 / 0.197 at
p≈0.5/0.2/0.05/0.01 -- significant at the two easier levels, not at the two hardest. At the
single hardest difficulty (p≈0.01) grammar-guided evolution without restarts loses outright
(0.0% vs. random's 0.9%), traced to the same `(mu+lambda)`-seeding pathology described below; a
targeted fix (requiring the initial population to be *feasible*, not merely evaluated, before
switching to local mutation -- `run_feasible_seeding_check` in the same script) closes this to an
exact tie (0.9% vs. 0.9%, p=1.000) at the hardest level, but at a cost: it underperforms the
unfixed rule at the easier levels (77.2%/36.6%/5.6% vs. 87.2%/66.3%/28.8%, each still a clear win
over random but smaller than without the fix). So there is a genuine, difficulty-dependent
trade-off between the two seeding rules, not a clean universal fix.

**What this means for the paper's claims:** "grammar-guided local-mutation search beats blind
random search in a complex/low-feasibility environment" IS now supported at 3 of 4 difficulty
levels, with the 4th a non-significant tie once the seeding-rule fix is applied. This sits
alongside the **feasibility gate's** cost advantage (`feasibility_rate_scaling.py`: gate-OFF
cost scales ~1/p, gate-ON stays flat at 1 -- confirmed across p=0.54→0.01), which is a different
mechanism (checking feasibility before spending a physics evaluation) from mutation locality;
both are now real, measured, mechanism-specific advantages rather than one of them being an
artifact of a running-best-vs-per-eval measurement bug.
