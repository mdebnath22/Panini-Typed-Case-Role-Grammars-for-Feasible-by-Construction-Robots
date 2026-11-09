#!/bin/bash
# Sol array job: RL-train one body per task against the freshly-rebuilt
# (post-geometry-fix) proxy-vs-RL catalog in results_refit/rank_corr_v2/.
#
# NEW — this is a clearly-separate script from any already-queued
# submit_search_final_designs_array.sh / submit_search_final_designs_priority_array.sh
# job the user already has in hand for results_refit/search21_final_designs(_priority)/.
# It does not touch those directories and can be submitted independently,
# before or after that job.
#
# Array size = number of geometrically_feasible bodies in
# results_refit/rank_corr_v2/manifest.json (31 as of this build: 28 from
# the randomly-sampled catalog + 3 feasible PĀṇini (panini_search, formerly
# grammar_guided_cem_v3; manifest "method" key left unchanged) final
# designs). Recompute N below if the manifest changes.
#
#SBATCH --job-name=rankcorr_v2
#SBATCH --partition=public
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --mem=16G
#SBATCH --array=0-30
#SBATCH --output=results_refit/rank_corr_v2/slurm_logs/%A_%a.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"
mkdir -p results_refit/rank_corr_v2/slurm_logs 2>/dev/null || true

PYBIN=/home/mroycho1/.conda/envs/mdebnath/bin/python3

"$PYBIN" -m experiments.rl_train_mjx \
    --out_dir results_refit/rank_corr_v2 \
    --body_index "${SLURM_ARRAY_TASK_ID}"

# After all array tasks finish:
#   python3 -m experiments.merge_rl_shards --out_dir results_refit/rank_corr_v2
#   python3 -m experiments.rank_correlation_analysis --out_dir results_refit/rank_corr_v2
