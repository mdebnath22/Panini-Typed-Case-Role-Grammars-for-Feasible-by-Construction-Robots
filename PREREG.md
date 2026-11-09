# Pre-registration: strengthening experiments

Written before any of the runs below. Every constant here is fixed now; any later change
will be recorded in a "Deviations" section at the bottom with the reason, never silently.

Shared: n = 8 limbs unless stated; difficulty levels are the pooled-calibrated taus already
in `results/cal_A_8.json` (pooled target pass rates 8%, 3%, 1%, 0.3%); feasibility = the
joint spec of `experiments/hard_coupled_calibrated.spec_tau`. Statistics: two-sided paired
Wilcoxon (search) or Fisher exact (counts), Holm correction across the difficulty levels of
each comparison family, 95% Wilson intervals for proportions, bootstrap 95% CIs for means.

## Exp 1 -- Feasibility-constrained task search

Question: does the grammar's feasibility advantage turn into better locomotion fitness when
infeasible designs may not be simulated?

* Levels: 8%, 1%, 0.3%.
* Rule: a free geometric gate runs before every physics call; a design that fails it is
  never simulated. Budgets per run: P = 2000 gate checks, B = 60 physics evaluations;
  a run stops at whichever is exhausted first.
* Fitness: `gmm.task_fitness.evaluate_fitness`, 500 steps, gait seed 0 during search.
  Reported number: the best feasible design of each run re-evaluated at 1000 steps,
  averaged over gait seeds 0-2. A run that finds no feasible design scores "none".
* Search scoring (all methods): feasible -> fitness; infeasible -> -1000 - violation
  (guides every method toward the feasible region; costs a gate check, not physics).
* (a) From scratch, 20 seeds: grammar (mu+lambda) ES over the 4-parameter genome;
  flat CMA-ES, flat PSO, flat random search (64-D); constraint-aware baselines of Exp 4.
* (b) Seeded, 20 seeds: every method starts from the same feasible symmetric body;
  grammar ES, flat per-field clip ES, range-constrained resampling ES (MJCF), flat CMA-ES.
* Primary outcome: best feasible fitness; primary test grammar vs. each baseline.

## Exp 2 -- Asymmetric payload environment and specificity ablation

Question: when feasible designs are not symmetric, does the grammar adapt through rule
arbitration where a fixed symmetric template cannot?

* Payload: 1.5 kg point mass at (+0.20, 0, 0) m from the torso centre, fixed to the torso.
  It enters the centre of mass (stability margin) and total mass; the mass budget is raised
  by the payload mass.
* New rule `B1.3_ADD_LIMB_LOAD`, specificity 2, consumes markers {has_torso, payload_task};
  the task sets `payload_task` as an initial marker (the same mechanism as the existing
  reach override B1.2). Its realization, fixed now from statics, not tuned:
  - mount angle theta' = theta + alpha * sin(phi - theta), alpha = 0.35 rad, phi = 0
    (limbs shift toward the payload side);
  - limbs with cos(theta' - phi) > 0 get every segment length scaled by s = 1.3.
  Otherwise identical to B1.1 (same exports).
* Conditions: flat per-limb; symmetric template (= grammar with B1.3 removed, identical
  construction); grammar with B1.3 (fires by arbitration, no code path change in the plan).
* Difficulty: tau re-calibrated for this environment on the pooled pass rate of all three
  samplers, targets 8%, 3%, 1%, 0.3%, same procedure as before.
* Outcomes: from-scratch pass rate (6000 samples each); CMA-ES from scratch (200 evals,
  20 seeds): seeds with a feasible design.
* A payload-aware hand-written template would reimplement B1.3 and match it by
  construction; this is stated, not run.

## Exp 3 -- Physics-grounded feasibility

Question: do the results hold when feasibility is checked in the simulator rather than by
geometric thresholds?

* Same payload body, attached as a welded MuJoCo body at the same offset.
* Checks on the compiled model, zero control, 400 settle steps:
  1. stands: torso height >= 0.12 m at the end and no torso-ground contact;
  2. no self-collision: no contact between two robot bodies that are not parent/child;
  3. effort: max |actuator force| over the last 100 steps <= 0.8 x the actuator's limit
     (gear x ctrl range), i.e. the pose is held with margin.
* Difficulty: payload mass in {0.5, 1.5, 3.0, 5.0} kg (heavier = rarer). No other knob.
* Conditions: flat per-limb; symmetric template; grammar with B1.3. 3000 samples each.

## Exp 4 -- Constraint-aware baselines

* CMA-ES with death penalty and resampling: infeasible proposals are redrawn (cost a
  gate check) until feasible or P is exhausted.
* Constrained Bayesian optimisation: GP regression on fitness of feasible points, GP
  classifier for feasibility, candidate chosen by EI x P(feasible) over 500 random
  candidates per step; flat 64-D space.
* A real LLM editor is not run: no LLM API is reachable from this workspace.

## Rigor

* Replicate the 15-condition hard-environment validity table at n = 10 (cal_A_10 levels).
* Holm correction and Wilson / bootstrap intervals for every new comparison.

## Deviations

1. (2026-10-19, after a 1-seed smoke test, before any full run) Exp 1 reported fitness:
   the evaluator returns a -1000 sentinel for a failed or blown-up rollout, and averaging
   it over three gait seeds gives meaningless values (e.g. -333). A failed rollout is now
   scored -0.3, the evaluator's own fall penalty with zero displacement. Applies to every
   method identically.
2. (same time) Exp 1 seeded flat CMA-ES: sigma was not fixed in this file; sigma = 0.05
   produced no feasible sample in the smoke test (all 64 coordinates move at once). Both
   sigma = 0.05 and sigma = 0.01 are run, and flat CMA-ES is reported at its better sigma.
3. (2026-10-19) Exp 1 cancelled before any full result. The compiler skips the torso's
   free joint, so every body is welded to the world and `evaluate_fitness`'s
   "displacement" is the first hip hinge angle. All locomotion-fitness results are
   invalid. Decision (author): drop fitness claims; the paper becomes feasibility-only.
4. Exp 3 needs a body that can fall, so it inserts a free joint on the torso inside its
   own compile step. The compiler used everywhere else is unchanged.
5. Exp 4 keeps its baselines (CMA-ES with resampling, constrained BO) but its objective
   becomes finding feasible designs from scratch (200 evaluations, 20 seeds), as in the
   hard-environment search, since there is no valid fitness to optimise.
6. (before any Exp 3 run) Exp 3 posture: with zero control every hip holds its leg
   horizontal, so every body would lie flat and the test would be degenerate. Fixed
   stance instead: every hip (limb-root) actuator commanded to +0.8 rad (leg 46 degrees
   below horizontal), every other actuator 0. Effort limit: the actuators are position
   servos with no force range, so "gear x ctrl range" is undefined; the limit is the servo's
   maximum force kp x 1 rad = 18 N m, and the check is max |force| <= 0.8 x 18.
7. Exp 3 cancelled after a 30-sample smoke test. With a free torso the compiled bodies are
   not physically sound: the torso ends below the floor (z = -0.20 m), hinge angles reach
   4-6 rad despite +/-0.9 rad limits, and limb-root capsules interpenetrate each other and
   the torso at the mount points. A physics-grounded check needs the body model rebuilt,
   which is outside the scope the author chose. The paper states that the simulator is used
   only to check that a model compiles and stays numerically well-formed with the torso fixed.
8. Exp 4 CMA-ES with death penalty: when every sample is infeasible the penalty is flat and
   CMA's flat-fitness termination stopped the run after a few evaluations. Corrected run
   (exp4b_cma_resample_fixed.py) restarts a stopped strategy until the 200-evaluation budget
   is used; the corrected numbers are reported, the original ones are not.
