#!/usr/bin/env bash
# Runs every CPU-side piece added for the GPU-handoff roadmap, in order.
# Run from the repository root: bash run_gpu_handoff_demo.sh
set -e
OUT=results_refit/rank_corr

echo "== 1/4: catalog + catalog_binding + engineering gate smoke test =="
python3 - <<'EOF'
from ptg.body_grammar import BodyGrammar, legged_plan
from ptg.body_sutra import seed_body_rules
from gmm.phi_body import PhiBody
from ptg.catalog_binding import bind_catalog
from sulba.feasibility import check_feasibility, FeasibilitySpec
from sulba.engineering_feasibility import check_engineering_feasibility, EngineeringSpec

g = BodyGrammar(seed_body_rules(), seed=0)
tree = PhiBody().build_tree(g.derive(legged_plan(n_legs=6, segments_per_leg=2)))
print("geometric feasible:", check_feasibility(tree, FeasibilitySpec()).ok)
bom = bind_catalog(tree)
print("BOM total cost: $%.2f, lead time: %d days" % (bom.total_cost, bom.lead_time_days))
eng = check_engineering_feasibility(tree, EngineeringSpec())
print("engineering feasible:", eng.ok, eng.checks)
EOF

echo
echo "== 2/4: MJCF+BOM compile and URDF export =="
python3 - <<'EOF'
from ptg.body_grammar import BodyGrammar, legged_plan
from ptg.body_sutra import seed_body_rules
from gmm.phi_body import PhiBody
from gmm.mjcf_compiler import compile_with_bom, compile_urdf

g = BodyGrammar(seed_body_rules(), seed=0)
tree = PhiBody().build_tree(g.derive(legged_plan(n_legs=6, segments_per_leg=2)))
out = compile_with_bom(tree, name="demo_body")
open("demo_body.xml", "w").write(out["xml"])
open("demo_body.urdf", "w").write(compile_urdf(tree, name="demo_body"))
print("wrote demo_body.xml, demo_body.urdf;", out["n_actuators"], "actuators")
EOF

echo
echo "== 3/4: rank-correlation harness (CPU half) =="
python3 -m experiments.rank_correlation_harness --n_bodies 40 --out_dir "$OUT"

echo
echo "== 4/4: rank-correlation analysis (will say it's waiting on rl_results.json) =="
python3 -m experiments.rank_correlation_analysis --out_dir "$OUT"

echo
echo "== task curricula (walk / parkour / waiter) =="
for t in walk parkour waiter; do
  python3 -m ptg.task_spec --task "$t" --out "docs/${t}_task_spec.json"
done

echo
echo "Done. Next: hand $OUT/manifest.json + $OUT/bodies/ to your GPU training run,"
echo "write results back to $OUT/rl_results.json, then re-run:"
echo "  python3 -m experiments.rank_correlation_analysis --out_dir $OUT"
