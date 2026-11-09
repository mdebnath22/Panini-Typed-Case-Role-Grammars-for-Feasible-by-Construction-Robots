#!/bin/bash
# Sol array job: RL-train one body per task for the PRIORITY manifest
# (results_refit/search21_final_designs_priority/manifest.json) -- 192
# bodies, 8 methods x 24 seeds, including cma_es_lite, 100% geometrically
# feasible. This is the smaller/faster run that covers the user's
# requested ours-vs-CMA-ES comparison.
#
# This is a SEPARATE job from submit_search_final_designs_array.sh (the
# full 672-body manifest) and from submit_rank_corr_v2_array.sh (the
# proxy-vs-RL correlation catalog) -- all three can be submitted
# independently, in any order.
#
# Array size = number of bodies in
# results_refit/search21_final_designs_priority/manifest.json (192 as of
# this build, all geometrically feasible). Recompute below if the
# manifest changes.
#
#SBATCH --job-name=search_priority
#SBATCH --partition=public
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --mem=16G
#SBATCH --array=0-191
#SBATCH --output=results_refit/search21_final_designs_priority/slurm_logs/%A_%a.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"
mkdir -p results_refit/search21_final_designs_priority/slurm_logs 2>/dev/null || true

PYBIN=/home/mroycho1/.conda/envs/mdebnath/bin/python3

"$PYBIN" -m experiments.rl_train_mjx \
    --out_dir results_refit/search21_final_designs_priority \
    --body_index "${SLURM_ARRAY_TASK_ID}"

# After all array tasks finish:
#   python3 -m experiments.merge_rl_shards --out_dir results_refit/search21_final_designs_priority
#   python3 -m experiments.search21_rl_rescoring_analysis --out_dir results_refit/search21_final_designs_priority --stored_json results_refit/search21_n24_b60.json
