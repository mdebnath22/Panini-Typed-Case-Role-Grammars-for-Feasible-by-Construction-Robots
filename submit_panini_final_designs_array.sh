#!/bin/bash
# Sol array job: RL-train one body per task for the PANINI manifest
# (results_refit/panini_final_designs/manifest.json) -- 144 bodies, 6
# methods (panini_search [PANINI, the proposed method -- THIS is the run
# that answers "does PANINI survive RL re-scoring", not GGE] x evolutionary_search
# [GGE, old "ours", kept for comparison] x ipop_cma_es x cma_es_lite x pso x
# cross_entropy_method) x 24 seeds, 100% geometrically feasible (independently
# re-validated: all 144 bodies load and step cleanly in MuJoCo with no NaN/inf).
#
# This is a SEPARATE job from submit_search_final_designs_array.sh,
# submit_search_final_designs_priority_array.sh and submit_rank_corr_v2_array.sh
# -- all can be submitted independently, in any order. Same established
# pattern as those scripts (--partition=public, SLURM_SUBMIT_DIR fallback,
# direct conda python binary path).
#
# Array size = number of bodies in
# results_refit/panini_final_designs/manifest.json (144 as of this build,
# all geometrically feasible). Recompute below if the manifest changes.
#
#SBATCH --job-name=panini_final
#SBATCH --partition=public
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --mem=16G
#SBATCH --array=0-143
#SBATCH --output=results_refit/panini_final_designs/slurm_logs/%A_%a.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"
mkdir -p results_refit/panini_final_designs/slurm_logs 2>/dev/null || true

PYBIN=/home/mroycho1/.conda/envs/mdebnath/bin/python3

"$PYBIN" -m experiments.rl_train_mjx \
    --out_dir results_refit/panini_final_designs \
    --body_index "${SLURM_ARRAY_TASK_ID}"

# After all array tasks finish:
#   python3 -m experiments.merge_rl_shards --out_dir results_refit/panini_final_designs
#   python3 -m experiments.search21_rl_rescoring_analysis --out_dir results_refit/panini_final_designs
#
# Note: search21_rl_rescoring_analysis.py's OURS_NAME constant is hardcoded to
# "grammar_guided_evolution (ours)" (the OLD "ours" string, i.e. GGE), not
# "panini_search (PANINI, proposed)". The script still runs unmodified and
# produces the full per-method mean-rank table + Friedman test correctly either
# way -- only its extra "ours vs rank X" highlighting lines at the end will not
# fire for PANINI. If you want those extra lines to highlight PANINI instead of
# GGE, edit that one constant locally before running (do not commit that edit
# over the original file's intended default without checking with the rest of
# the project, since other manifests still rely on the GGE meaning of OURS_NAME).
