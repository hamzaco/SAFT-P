#!/bin/bash
#SBATCH -J cube_scan_bethe
#SBATCH -o cube_scan_bethe.%j.out
#SBATCH -e cube_scan_bethe.%j.err
#SBATCH -p volta-cpu
#SBATCH -N 1
#SBATCH --ntasks-per-node=50
#SBATCH -w node[200-203]

set -euo pipefail

source /fast/shared/anaconda/2025.12/etc/profile.d/conda.sh
conda activate myenv

# Pin all threading to 1.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export BLIS_NUM_THREADS=1
export NUMBA_NUM_THREADS=1
export NUMBA_THREADING_LAYER=workqueue

# Reduce glibc heap retention.
export MALLOC_MMAP_THRESHOLD_=262144
export MALLOC_TRIM_THRESHOLD_=131072
export MALLOC_MMAP_MAX_=65536

ROOT_DIR="${SLURM_SUBMIT_DIR}"
SCRIPT="${ROOT_DIR}/cube_spinodal_scan_representative_assoc_memory_nu_fixed.py"

CASE_TAG="${CASE_TAG:-cube_case}"
PATCHES_NPY="${PATCHES_NPY:-${ROOT_DIR}/${CASE_TAG}_patches.npy}"
CACHE_PATH="${CACHE_PATH:-${ROOT_DIR}/cache/${CASE_TAG}_cache_streaming_directional_dir}"
WORKDIR="${WORKDIR:-/pool/hamza/${CASE_TAG}_scan_selfconsistent_bethe}"
SHARD_DIR="${SHARD_DIR:-${WORKDIR}/shards}"
FINAL_NPZ="${FINAL_NPZ:-${WORKDIR}/${CASE_TAG}_selfconsistent_stateassoc_spinodal_scan.npz}"
MMAP_DIR="${MMAP_DIR:-${WORKDIR}/cache_mmap_representative_${CASE_TAG}_${SLURM_JOB_ID:-manual}}"

EPS_A_MIN="${EPS_A_MIN:-0.0}"
EPS_A_MAX="${EPS_A_MAX:-4.0}"
N_EPS_A="${N_EPS_A:-11}"
EPS_C_MIN="${EPS_C_MIN:-0.0}"
EPS_C_MAX="${EPS_C_MAX:-1.2}"
N_EPS_C="${N_EPS_C:-13}"
PHI_MIN="${PHI_MIN:-0.01}"
PHI_MAX="${PHI_MAX:-0.99}"
N_PHIS="${N_PHIS:-101}"

# Keep the representative baseline unchanged. Only the explicit Bethe term differs.
FACTOR="${FACTOR:-6.0}"
LINEAR_TERM_MODE="${LINEAR_TERM_MODE:-logZnu}"
REGISTRY_MODE="${REGISTRY_MODE:-boltzmann}"
ASSOCIATION_MODE="${ASSOCIATION_MODE:-oriented_directed_face_state}"
USE_BOUNDARY_QUOTIENT="${USE_BOUNDARY_QUOTIENT:-0}"

TOL_SOLVER="${TOL_SOLVER:-1e-5}"
ACCEPT_RESIDUAL="${ACCEPT_RESIDUAL:-1e-5}"
MAX_ITER_SOLVER="${MAX_ITER_SOLVER:-250}"
MAX_JAC_REBUILDS="${MAX_JAC_REBUILDS:-6}"
FALLBACK_NEWTON="${FALLBACK_NEWTON:-1}"

# Optional post-solve boundary-state Bethe compatibility correction.
# BETHE_STRENGTH=0 or BETHE_CORRECTION=none must reproduce the representative baseline.
BETHE_CORRECTION="${BETHE_CORRECTION:-none}"
BETHE_STRENGTH="${BETHE_STRENGTH:-0.0}"
BETHE_CONTACT_FACTOR="${BETHE_CONTACT_FACTOR:-1.0}"
BETHE_COMPAT_MODE="${BETHE_COMPAT_MODE:-attractive}"
BETHE_ORIENTATION_MODE="${BETHE_ORIENTATION_MODE:-orbit_average}"
BETHE_THRESHOLD="${BETHE_THRESHOLD:-1e-8}"
BETHE_MAX_ITER="${BETHE_MAX_ITER:-500}"
BETHE_TOL="${BETHE_TOL:-1e-8}"
BETHE_SC_MAX_ITER="${BETHE_SC_MAX_ITER:-20}"
BETHE_SC_TOL="${BETHE_SC_TOL:-1e-5}"
BETHE_SC_DAMPING="${BETHE_SC_DAMPING:-0.5}"

mkdir -p "$WORKDIR" "$SHARD_DIR" "$MMAP_DIR" "$(dirname "$CACHE_PATH")"
cd "$ROOT_DIR"

echo "=================================================="
echo "job id         : $SLURM_JOB_ID"
echo "partition      : $SLURM_JOB_PARTITION"
echo "nodelist       : $SLURM_JOB_NODELIST"
echo "nodes          : $SLURM_JOB_NUM_NODES"
echo "ntasks         : $SLURM_NTASKS"
echo "submit dir     : $ROOT_DIR"
echo "script         : $SCRIPT"
echo "case tag       : $CASE_TAG"
echo "patches file   : $PATCHES_NPY"
echo "cache path     : $CACHE_PATH"
echo "mmap dir       : $MMAP_DIR"
echo "shard dir      : $SHARD_DIR"
echo "final npz      : $FINAL_NPZ"
echo "grid           : ${N_EPS_A} x ${N_EPS_C}"
echo "phis           : ${N_PHIS}"
echo "closure        : representative baseline + optional post-solve Bethe diagnostic correction"
echo "linear term    : ${LINEAR_TERM_MODE}"
echo "registry mode  : ${REGISTRY_MODE}"
echo "association mode: ${ASSOCIATION_MODE}"
echo "factor         : ${FACTOR}"
echo "boundary quotient: ${USE_BOUNDARY_QUOTIENT}"
echo "fallback Newton : ${FALLBACK_NEWTON}"
echo "bethe correction : ${BETHE_CORRECTION}"
echo "bethe strength   : ${BETHE_STRENGTH}"
echo "bethe contact factor: ${BETHE_CONTACT_FACTOR}"
echo "bethe compat mode: ${BETHE_COMPAT_MODE}"
echo "bethe orientation: ${BETHE_ORIENTATION_MODE}"
echo "bethe threshold  : ${BETHE_THRESHOLD}"
echo "bethe max iter   : ${BETHE_MAX_ITER}"
echo "bethe tol        : ${BETHE_TOL}"
echo "bethe sc max iter: ${BETHE_SC_MAX_ITER}"
echo "bethe sc tol     : ${BETHE_SC_TOL}"
echo "bethe sc damping : ${BETHE_SC_DAMPING}"
echo "=================================================="

[[ -f "$PATCHES_NPY" ]] || { echo "ERROR: patches file not found: $PATCHES_NPY"; exit 2; }
[[ -e "$CACHE_PATH" ]] || { echo "ERROR: cache path not found: $CACHE_PATH"; exit 2; }
[[ -f "$SCRIPT" ]] || { echo "ERROR: script file not found: $SCRIPT"; exit 2; }

echo "[prepare] unpacking cache into mmap format (with bond_hist)..."
python "$SCRIPT" prepare-mmap \
  --cache-path "$CACHE_PATH" \
  --mmap-dir "$MMAP_DIR"

for f in cfg.npy bond_hist.npy species_counts.npy group_keys.npy group_ptr.npy meta.npz .ready; do
  [[ -e "$MMAP_DIR/$f" ]] || { echo "ERROR: missing mmap artifact $MMAP_DIR/$f"; exit 2; }
done

LOCAL_MMAP="/tmp/${CASE_TAG}_${SLURM_JOB_ID}_mmap_cache_representative_bethe"

echo "[stage] copying mmap cache to local disk on each node..."
srun --kill-on-bad-exit=1 --ntasks-per-node=1 --ntasks=$SLURM_JOB_NUM_NODES bash -c '
  set -euo pipefail
  LOCAL="'"$LOCAL_MMAP"'"
  SRC="'"$MMAP_DIR"'"
  mkdir -p "$LOCAL"
  cp -v "$SRC"/*.npy "$SRC"/meta.npz "$SRC"/.ready "$LOCAL"/
  echo "[$(hostname)] copied mmap cache to $LOCAL ($(du -sh "$LOCAL" | cut -f1))"
'

echo "[stage] run shards"
srun --kill-on-bad-exit=1 --ntasks=$SLURM_NTASKS --cpus-per-task=1 bash -lc '
set -euo pipefail
cd "'"$ROOT_DIR"'"
python "'"$SCRIPT"'" run-shard \
  --patches "'"$PATCHES_NPY"'" \
  --cache-path "'"$CACHE_PATH"'" \
  --mmap-dir "'"$LOCAL_MMAP"'" \
  --out-dir "'"$SHARD_DIR"'" \
  --shard-id "$SLURM_PROCID" \
  --n-shards "'"$SLURM_NTASKS"'" \
  --eps-a-min "'"$EPS_A_MIN"'" \
  --eps-a-max "'"$EPS_A_MAX"'" \
  --n-eps-a "'"$N_EPS_A"'" \
  --eps-c-min "'"$EPS_C_MIN"'" \
  --eps-c-max "'"$EPS_C_MAX"'" \
  --n-eps-c "'"$N_EPS_C"'" \
  --phi-min "'"$PHI_MIN"'" \
  --phi-max "'"$PHI_MAX"'" \
  --n-phis "'"$N_PHIS"'" \
  --factor "'"$FACTOR"'" \
  --linear-term-mode "'"$LINEAR_TERM_MODE"'" \
  --registry-mode "'"$REGISTRY_MODE"'" \
  --association-mode "'"$ASSOCIATION_MODE"'" \
  --bethe-correction "'"$BETHE_CORRECTION"'" \
  --bethe-strength "'"$BETHE_STRENGTH"'" \
  --bethe-contact-factor "'"$BETHE_CONTACT_FACTOR"'" \
  --bethe-compat-mode "'"$BETHE_COMPAT_MODE"'" \
  --bethe-orientation-mode "'"$BETHE_ORIENTATION_MODE"'" \
  --bethe-threshold "'"$BETHE_THRESHOLD"'" \
  --bethe-max-iter "'"$BETHE_MAX_ITER"'" \
  --bethe-tol "'"$BETHE_TOL"'" \
  --bethe-sc-max-iter "'"$BETHE_SC_MAX_ITER"'" \
  --bethe-sc-tol "'"$BETHE_SC_TOL"'" \
  --bethe-sc-damping "'"$BETHE_SC_DAMPING"'" \
  --tol-solver "'"$TOL_SOLVER"'" \
  --accept-residual "'"$ACCEPT_RESIDUAL"'" \
  --max-iter-solver "'"$MAX_ITER_SOLVER"'" \
  --max-jac-rebuilds "'"$MAX_JAC_REBUILDS"'" \
  $( [[ "'"$USE_BOUNDARY_QUOTIENT"'" == "1" ]] && echo --use-boundary-quotient || echo --no-boundary-quotient ) \
  $( [[ "'"$FALLBACK_NEWTON"'" == "1" ]] && echo --fallback-newton )
'

echo "[cleanup] removing local mmap caches..."
srun --kill-on-bad-exit=0 --ntasks-per-node=1 --ntasks=$SLURM_JOB_NUM_NODES bash -c '
  rm -rf "'"$LOCAL_MMAP"'"
  echo "[$(hostname)] cleaned up local mmap cache"
'

echo "[merge] assemble final scan"
python "$SCRIPT" merge-results \
  --out-dir "$SHARD_DIR" \
  --final-npz "$FINAL_NPZ" \
  --eps-a-min "$EPS_A_MIN" \
  --eps-a-max "$EPS_A_MAX" \
  --n-eps-a "$N_EPS_A" \
  --eps-c-min "$EPS_C_MIN" \
  --eps-c-max "$EPS_C_MAX" \
  --n-eps-c "$N_EPS_C" \
  --phi-min "$PHI_MIN" \
  --phi-max "$PHI_MAX" \
  --n-phis "$N_PHIS" \
  --factor "$FACTOR" \
  --linear-term-mode "$LINEAR_TERM_MODE" \
  --registry-mode "$REGISTRY_MODE" \
  --association-mode "$ASSOCIATION_MODE" \
  --bethe-correction "$BETHE_CORRECTION" \
  --bethe-strength "$BETHE_STRENGTH" \
  --bethe-contact-factor "$BETHE_CONTACT_FACTOR" \
  --bethe-compat-mode "$BETHE_COMPAT_MODE" \
  --bethe-orientation-mode "$BETHE_ORIENTATION_MODE" \
  --bethe-threshold "$BETHE_THRESHOLD" \
  --bethe-max-iter "$BETHE_MAX_ITER" \
  --bethe-tol "$BETHE_TOL" \
  --bethe-sc-max-iter "$BETHE_SC_MAX_ITER" \
  --bethe-sc-tol "$BETHE_SC_TOL" \
  --bethe-sc-damping "$BETHE_SC_DAMPING" \
  $( [[ "$USE_BOUNDARY_QUOTIENT" == "1" ]] && echo --use-boundary-quotient || echo --no-boundary-quotient )

echo "DONE: $FINAL_NPZ"
