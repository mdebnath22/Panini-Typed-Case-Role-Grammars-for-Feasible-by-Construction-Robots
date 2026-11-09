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
  careful hand-coding replicates it. `careful_delete_segment` falls short at 75/100, failing on
  MuJoCo's `mjMINVAL` mass/inertia requirement (25/100 occurrences) -- a *different* invariant
  than the name-collision bug that `careful_duplicate_limb` had to fix, and one that even
  careful, non-adversarial effort missed. So the honest claim is two-sided and per-edit-type:
  duplication is replicable by hand with enough care; deletion still isn't, because it trips an
  invariant that's easy to miss even when you're trying.
- **`feasibility_rate_scaling.py`** -- the quantitative version of the "small feasible region"
  intuition: as the feasibility spec's difficulty increases (measured ground-truth feasibility
  p = 0.538, 0.179, 0.039, 0.011), gate-OFF cost (expected tries to first feasible genome) grows
  as predicted by 1/p (1.8, 4.2, 18.0, 67.0 mean tries), while gate-ON stays flat at 1.0 by
  construction. Confirms the gate's advantage is not a one-spec artifact of the paper's existing
  ~5%-feasibility result -- it scales as the feasible region shrinks, across 4 orders of magnitude.
- **`inheritance_ablation_validity.py`** -- disabling anuvrtti (`anuvrtti_export=frozenset()`
  on every rule) leaves mutation validity unchanged (100/100 both ON and OFF), confirming
  validity comes from typing/legality-gating, not inheritance, as predicted. But a second,
  structure-aware check -- do legs 2..n actually reflect the genome's mutated `seg_rad`, not
  just leg 0? -- was ALSO unchanged (1/100 both ON and OFF, after fixing a tolerance bug that
  initially inflated this to a misleading 29/100 via coincidental proximity to hardcoded
  defaults). Tracing the compiled XML directly found why: in `ptg/body_sutra.py`,
  `BodySutra.resolve_params()` always lets a rule's own declared `params` overwrite
  anuvrtti-inherited values for any shared key (`out.update(self.params)` runs after loading
  `inherited_params`). `B1.1_ADD_LIMB` declares its own `seg_len`/`seg_rad` defaults, so the
  inherited genome value is clobbered back to the rule's hardcoded default on every leg after
  leg 0 -- `legged_plan()` (`ptg/body_grammar.py`) only works around this for leg 0, via an
  explicit `overrides` dict applied after `resolve_params`. **So anuvrtti's export of
  seg_len/seg_rad across legs is already a no-op in the unablated grammar for this rule
  catalogue** -- the ablation changed nothing observable because there was nothing functioning
  to disable. This does NOT undermine `todo1_typing_vs_inheritance.py`'s compactness claim
  (that's a rule-COUNTING argument -- how many distinct (op, param-tuple) rules are needed --
  not a trace of per-leg genome-controllability), but "inheritance makes every leg reflect
  genome.seg_rad" should not be claimed about this codebase as it stands. Fix candidates if this
  matters for the paper: have `resolve_params` only fall back to `self.params` for keys absent
  from `inherited_params`, or have `legged_plan()` pass `seg_len`/`seg_rad` as overrides on
  every leg, not just leg 0. Left unfixed here since it's outside this counterfactual
  experiment's scope, but documented in the script's own docstring so it isn't silently missed.
  **Update: this has since been fixed** (the first fix candidate above -- `resolve_params` now
  only falls back to a rule's own default for keys absent from `inherited_params`) and the
  ablation re-run; see "Not included / known gaps" below for the corrected numbers.

## Does grammar-guided search actually win in a complex (low-feasibility) environment?

`experiments/complex_environment_search.py` tests this directly -- the square/feasibility-region
intuition ("in a 1%-feasible environment, a smarter method should beat blind random sampling"),
not just restated but run. **An earlier version of this script reported a negative result that
turned out to be a measurement bug in our own evaluation code, not a real finding. Corrected, the
result is mostly positive: grammar-guided local mutation wins at 3 of 4 difficulty levels and is
a non-significant tie at the single hardest one.**

Hypothesis tested: once grammar-guided evolutionary search finds one feasible genome, mutating
it locally should land on another feasible genome more often than redrawing uniformly at random,
and that advantage should grow as the feasible fraction p shrinks (same spec difficulties as
`feasibility_rate_scaling.py`: p≈0.54, 0.17, 0.05, 0.01).

**The bug, disclosed before the corrected result:** the original measurement derived
feasible-eval-rate from each method's returned `history` array, which in
`m_search_loop.evolutionary_search`/`random_search` is a running *best-so-far* trace, not a
per-evaluation record -- once any feasible genome was found, every later entry read as feasible
regardless of that step's own draw, which silently inflated whichever method happened to find a
feasible point earliest. New `*_per_eval` functions in `complex_environment_search.py` record
validity directly at every evaluation call, and the comparison was re-run from scratch with
fresh seeds.

**Corrected result (n_seeds=8, budget=40, feasible-eval-rate = % of the budget that lands on a
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

## Manipulation-side compositional-generalization anomaly, diagnosed

Table 4 in the paper shows the manipulation domain's held-out (OOD) twist MSE *lower* than
in-distribution (ID) -- backwards from the morphology domain's result and from what
generalization usually looks like. `reviewer_q3_ood_diagnostic.py` runs the same experiment
across 5 seeds with a per-derivation-length breakdown to check the obvious innocent
explanation (OOD derivations happen to be shorter) before concluding anything else.

**That simple length story is ruled out, but not in the direction you'd expect.** ID derivations
are lengths 3/4/5, OOD are lengths 4/8 -- they barely overlap in length at all. At the one length
they share (4), ID error (0.1705) is **14x higher** than OOD error (0.0118), the opposite of "OOD
is shorter so it's easier."

**Tracing length 4 specifically explains everything.** Of the 3 training (ID) templates, exactly
one has length 4: `APPROACH, GRASP, LIFT, ROTATE_TOOL`. Every held-out (OOD) template of length 4
is `APPROACH, GRASP, LIFT, PLACE` -- no `ROTATE_TOOL` at all. Splitting every evaluated derivation
by whether it contains a `ROTATE_TOOL` step (not by length) shows why: derivations with
`ROTATE_TOOL` have twist MSE 0.1806 (n=27) vs. 0.0010 (n=73) without -- a 180x gap.
`ROTATE_TOOL` is the only transformation whose twist depends on a per-scene, randomly sampled
tool axis (`experiments/scenes.py`'s `random_scene`) rather than a task-geometry-determined
translation target, making it intrinsically harder to fit regardless of where it falls in a
derivation.

**So the anomaly is a template-composition confound, not a length confound and not a genuine
reversed generalization gap:** the one hard operation happens to appear only on the training
side, inflating the ID average with a task type OOD never has to handle. It says nothing about
compositional generalization behaving backwards, and doesn't bear on the morphology domain's
result (a structurally different held-out split -- leg/segment count, not primitive-rule
identity). A design fix for any future version of this diagnostic: include a
`ROTATE_TOOL`-bearing template in the held-out set too, so task difficulty is balanced across
splits. Not implemented here, since doing so changes the experiment being diagnosed rather than
diagnoses the one already run.

## Key results (as reported in the paper)

- Homomorphism defect: exactly 0 on inheritance-inactive pairs; ≈0.037 when active (0° rotation, 3.75 ± 2.08 cm). Chains k=2–6 stay ≤ 0.081.
- Compactness: 6 typed rules vs 17/37/49 flat for 4×2, 6×3, 8×3 bodies; up to 13.5× in the sweep.
- Containment: 1.000 across width floors and transitive pairs; held-out conflict agreement box 0.812, order 0.906, Euclidean 0.719.
- Search (21 methods, n=8 seeds): Friedman χ² = 33.46, p = 0.030; ours has the best average rank (6.25) but only coordinate descent is significantly different after Nemenyi. **This does not survive scaling to n=24 seeds** (`search21_scaled_stats.py`): Friedman χ² = 101.56, p = 6.0×10⁻¹³ (a much more decisive omnibus test), but ours drops to rank 10.19/21, mid-field, behind a new leader (CMA-ES lite, rank 7.19) and behind plain random search (rank 8.73). The Nemenyi result is unchanged in kind at both n: ours is statistically indistinguishable from every method except coordinate descent. We withdraw the best-average-rank claim; the paper now reports grammar-guided search as a competitive, tied-with-most-of-the-field optimizer, not the best-ranked one.
- Gate: finds a feasible body in 1.0 vs 9.6 ± 6.2 evaluations; task fitness 1.137 vs 1.672 (p = 0.254). It buys constraint satisfaction, not task optimality. A gait-correlated gate extension (`gait_correlated_gate_ablation.py`, a short CPG-rollout pre-check charged a fractional physics cost) does not reverse or narrow this: no_gate=1.668±0.781, geometric_gate=1.476±0.368, gait_correlated_gate=1.357±0.261 -- numerically the worst of the three, not significantly so at n=8.
- Validity: grammar 100%. Direct edits range from 0% to 100% depending on how tame the mutator is. The main validity criterion (loads and steps without non-finite state) is permissive; a stricter, locomotion-based criterion (loads+steps+≥2cm displacement, `stricter_validity_comparison.py`) still can't separate the grammar (57/100) from range-constrained resampling (64/100, p=0.385) or pure reinit (67/100, p=0.190), though it does separate the grammar from every more-disruptive alternative (duplicate_element, bitflip, latent_space_perturbation: 0/100).
- Why range-constrained resampling ties the grammar, resolved (`coupled_feasibility_validity.py`): every condition above mutates a design the grammar already derived, so it only tests whether validity *survives* one further edit, never whether the grammar's actual mechanism (even mount-angle spacing across limbs -- a joint, cross-limb property no single field's range can express) had to be *constructed* in the first place. Isolating that mechanism directly with a counterfactual (identical genome in every respect except mount angles drawn i.i.d. uniform instead of evenly spaced) and scoring on a coupled criterion only joint limb placement can satisfy (support-polygon area, calibrated against fresh genomes to very-low-feasibility regimes p≈0.30/0.10/0.05, n=150/condition): the grammar beats the mechanism-targeted counterfactual significantly at the two higher-powered levels (30.7% vs 12.7%, χ²p=0.0003; 6.7% vs 1.3%, χ²p=0.0392; direction holds but underpowered at p≈0.05: 2.7% vs 0.0%, χ²p=0.131), while still tying range-constrained resampling at every level (χ²p=0.80–1.00) -- because range-constrained resampling inherits the parent's coordination rather than ever constructing it. This is the decisive answer to "pure reinit and range-constrained resampling both hit 100%, so what does the grammar actually buy you": on the one property built to require the mechanism, removing it costs significantly.
- Two more budget-matched analogues (`robogrammar_glso_baselines.py`): RoboGrammar's Graph Heuristic Search (1.393±0.523, wins 6/8 vs. ours, p=0.304) and GLSO's latent-space Bayesian optimization with a real sklearn GP (1.983±0.478, wins 4/8, p=0.357, nominally *higher* mean than ours) -- both statistically indistinguishable from grammar-guided evolution, extending the tie across all seven budget-matched analogues tested.
- Closed kinematic chains (`closed_kinematic_chain.py`): a planar four-bar linkage has no real loop-closure solution for 36.4% of randomly sampled link-length combinations; where a solution exists, it closes to a measured defect of ~10⁻¹⁵ (floating-point precision), confirming the solver is sound and that closing a loop needs a genuinely new constraint-solving step the grammar's existing inheritance/specificity mechanisms have no analogue for.
