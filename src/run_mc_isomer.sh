#!/bin/bash
#SBATCH -J mc_isomer
#SBATCH -o logs/mc_isomer_%A_%a.out
#SBATCH -e logs/mc_isomer_%A_%a.err
# Array task for the revised Fig. 7 isomer Monte Carlo campaign.
# Manifest line: MODE T L MU1 SEED
#   MODE=wl          : MU1 is the sampling field; coexistence is found by reweighting.
#   MODE=branch      : MU1 is the fixed diagnostic field.
#   MODE=branch-coex : MU1 is a placeholder; the code reads mu_coex from WL_SOURCE.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKDIR="${WORKDIR:?WORKDIR not set}"
MANIFEST="${MANIFEST:?MANIFEST not set}"
WL_SOURCE="${WL_SOURCE:-$WORKDIR}"

source /fast/shared/anaconda/2025.12/etc/profile.d/conda.sh
conda activate myenv

export OMP_NUM_THREADS=1
export NUMBA_CACHE_DIR="${WORKDIR}/numba_cache"
mkdir -p "$NUMBA_CACHE_DIR"

read -r MODE T L MU1 SEED < <(sed -n "${SLURM_ARRAY_TASK_ID}p" "$MANIFEST")
echo "task ${SLURM_ARRAY_TASK_ID}: mode=$MODE T=$T L=$L mu1=$MU1 seed=$SEED on $(hostname)"

# Seeded branch-chain length scales as L^2: SWEEPS means attempted moves/site.
STEPS=$(awk -v l="$L" -v s="${SWEEPS:-500000}" 'BEGIN{printf "%.0f", s*l*l}')

case "$MODE" in
  branch)
    python "${ROOT_DIR}/mc_isomer_cluster_coex.py" --mode branch \
      --T "$T" --L "$L" --mu1 "$MU1" --seed "$SEED" \
      --steps "$STEPS" --interval "${INTERVAL:-50000}" --outdir "$WORKDIR"
    ;;

  branch-coex)
    python "${ROOT_DIR}/mc_isomer_cluster_coex.py" --mode branch-coex \
      --T "$T" --L "$L" --wl-source "$WL_SOURCE" --seed "$SEED" \
      --steps "$STEPS" --interval "${INTERVAL:-50000}" --outdir "$WORKDIR"
    ;;

  wl)
    python "${ROOT_DIR}/mc_isomer_cluster_coex.py" --mode wl \
      --T "$T" --L "$L" --mu1 "$MU1" --seed "$SEED" \
      --wl-fmin "${WL_FMIN:-1e-6}" --wl-flat "${WL_FLAT:-0.8}" \
      --wl-min-steps "${WL_MIN_STEPS:-4e6}" --wl-max-steps "${WL_MAX_STEPS:-2e9}" \
      --min-barrier "${MIN_BARRIER:-0.5}" --outdir "$WORKDIR"
    ;;

  *)
    echo "Unknown mode: $MODE" >&2
    exit 2
    ;;
esac
