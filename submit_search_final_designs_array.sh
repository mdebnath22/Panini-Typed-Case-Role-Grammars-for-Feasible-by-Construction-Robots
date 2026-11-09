#!/bin/bash
# Sol array job: RL-train one body per task for the FULL manifest
# (results_refit/search21_final_designs/manifest.json) -- 672 bodies,
# all 28 methods (21 original + 7 recent-SOTA analogues) x 24 seeds,
# 100% geometrically feasible.
#
# This is a SEPARATE job from submit_search_final_designs_priority_array.sh
# (the smaller 192-body priority subset) and from
# submit_rank_corr_v2_array.sh (the proxy-vs-RL correlation catalog) --
# all three can be submitted independently, in any order. This is the
# larger, slower run (~168 GPU-hours estimated) -- run the priority
# array first if you want the ours-vs-CMA-ES answer quickly.
#
# Array size = number of bodies in
# results_refit/search21_final_designs/manifest.json (672 as of this
# build, all geometrically feasible). Recompute below if the manifest
# changes.
#
#SBATCH --job-name=search_full
#SBATCH --partition=public
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --mem=16G
#SBATCH --array=0-671
#SBATCH --output=results_refit/search21_final_designs/slurm_logs/%A_%a.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"
mkdir -p results_refit/search21_final_designs/slurm_logs 2>/dev/null || true

PYBIN=/home/mroycho1/.conda/envs/mdebnath/bin/python3

"$PYBIN" -m experiments.rl_train_mjx \
    --out_dir results_refit/search21_final_designs \
    --body_index "${SLURM_ARRAY_TASK_ID}"

# After all array tasks finish:
#   python3 -m experiments.merge_rl_shards --out_dir results_refit/search21_final_designs
#   python3 -m experiments.search21_rl_rescoring_analysis --out_dir results_refit/search21_final_designs --stored_json results_refit/search21_n24_b60.json
