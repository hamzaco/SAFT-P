#!/bin/bash
#SBATCH -J mc_line
#SBATCH -o logs/mc_line.%A_%a.out
#SBATCH -e logs/mc_line.%A_%a.err
#SBATCH -p volta-cpu
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
# No --mem: let the partition default apply.  Override per-submission with
#   MEM=16G bash submit_mc_line.sh
#SBATCH --time=1-00:00:00
#
# One segment of the critical line eps_nd,c(eps_d) at one lattice size, for the
# stick and L-shaped patchy-particle models.  This is the notebook's
# continuation sweep (notebooks/mc_sims_{l,stick}_shaped.ipynb, main()) run at
# L = 10, 20, 30 so the whole line can be extrapolated to L -> infinity.
#
# Do NOT sbatch this file directly -- use submit_mc_line.sh, which builds the
# per-L manifests and sets the per-L walltime and array range.  Direct
# submission works but every task then books the L=30 walltime.
#
# Required environment (submit_mc_line.sh sets these):
#   MANIFEST   task manifest, one "system L seg lo hi eps_nd0 mu0 s0 seed"/line
#   WORKDIR    output directory
# Optional:
#   SWEEPS_REF REPLICAS SAMPLES SCOUT_FRAC EPS_D_STEP INIT SCAN_FALLBACK
#   MAX_RESUBMIT RESUBMIT_COUNT EXTRA_ARGS

set -euo pipefail

source /fast/shared/anaconda/2025.12/etc/profile.d/conda.sh
conda activate myenv

# One thread per process: the MC kernel is serial numba and we parallelise
# over replicas with multiprocessing, not over BLAS.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export BLIS_NUM_THREADS=1
export NUMBA_NUM_THREADS=1
export NUMBA_THREADING_LAYER=workqueue

ROOT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
SCRIPT="${ROOT_DIR}/mc_line_sweep.py"

WORKDIR="${WORKDIR:-/pool/hamza/mc_line}"
MANIFEST="${MANIFEST:-${WORKDIR}/manifest.txt}"
ISING_CACHE="${ISING_CACHE:-${WORKDIR}/ising_ref.npz}"

SWEEPS_REF="${SWEEPS_REF:-6e6}"
REPLICAS="${REPLICAS:-${SLURM_CPUS_PER_TASK:-8}}"
SAMPLES="${SAMPLES:-6000}"
BURN_FRAC="${BURN_FRAC:-0.10}"
SCOUT_FRAC="${SCOUT_FRAC:-0.08}"
EPS_D_STEP="${EPS_D_STEP:-0.02}"
INIT="${INIT:-middle}"
SCAN_FALLBACK="${SCAN_FALLBACK:-1}"
MAX_RESUBMIT="${MAX_RESUBMIT:-8}"
RESUBMIT_COUNT="${RESUBMIT_COUNT:-0}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

# Numba's on-disk function cache: keep it node-local so array tasks that land
# on different nodes do not fight over the same files on shared storage.
export NUMBA_CACHE_DIR="/tmp/numba_cache_${USER:-$(id -un)}_${SLURM_JOB_ID:-manual}"
mkdir -p "$NUMBA_CACHE_DIR"

mkdir -p "$WORKDIR" "${ROOT_DIR}/logs"
cd "$ROOT_DIR"

[[ -f "$SCRIPT" ]]   || { echo "ERROR: missing $SCRIPT"; exit 2; }
[[ -f "$MANIFEST" ]] || { echo "ERROR: missing manifest $MANIFEST"; exit 2; }

TASK_ID="${SLURM_ARRAY_TASK_ID:-1}"
LINE=$(sed -n "${TASK_ID}p" "$MANIFEST")
[[ -n "$LINE" ]] || { echo "ERROR: manifest line $TASK_ID is empty"; exit 2; }
read -r SYSTEM LSIDE SEG LO HI EPS_ND0 MU0 S0 SEED <<< "$LINE"

TAG="${SYSTEM}_L${LSIDE}_seg${SEG}_s${SEED}"
OUT="${WORKDIR}/${TAG}.npz"
CKPT="${WORKDIR}/ck_${TAG}.json"

echo "=================================================="
echo "job id       : ${SLURM_JOB_ID:-manual}  array task ${TASK_ID}"
echo "partition    : ${SLURM_JOB_PARTITION:-?}"
echo "node         : $(hostname)"
echo "cpus-per-task: ${SLURM_CPUS_PER_TASK:-?}"
echo "submit dir   : $ROOT_DIR"
echo "workdir      : $WORKDIR"
echo "manifest     : $MANIFEST (line $TASK_ID -> '$LINE')"
echo "system       : $SYSTEM"
echo "L            : $LSIDE"
echo "segment      : $SEG   eps_d $LO -> $HI  step $EPS_D_STEP"
echo "anchor       : eps_nd=$EPS_ND0  mu=$MU0  s=$S0"
echo "seed         : $SEED"
echo "sweeps_ref   : $SWEEPS_REF   (sweeps at L=10; scaled by (L/10)^2.17)"
echo "replicas     : $REPLICAS"
echo "scout_frac   : $SCOUT_FRAC"
echo "init         : $INIT"
echo "checkpoint   : $CKPT"
echo "resubmit     : $RESUBMIT_COUNT/$MAX_RESUBMIT"
echo "out          : $OUT"
echo "started      : $(date)"
echo "=================================================="

if [[ -s "$OUT" && "${FORCE:-0}" != "1" ]]; then
  echo "[skip] $OUT already exists (set FORCE=1 to overwrite)"
  exit 0
fi

# FORCE means "recompute this point".  The checkpoint has to go with the .npz:
# leaving it behind makes the next run resume a completed grid, skip the loop
# body entirely, and write the OLD points straight back out -- a re-run that
# silently returns the previous answer.  (mc_line_core also refuses to resume a
# checkpoint whose fingerprint has changed; this is the belt to that braces.)
if [[ "${FORCE:-0}" == "1" && -e "$CKPT" ]]; then
  echo "[force] removing stale checkpoint $CKPT"
  rm -f "$CKPT"
fi

# The Ising reference is shared by every task.  Build it once, under a lock,
# so that a whole array starting together does not race on the same file.
if [[ ! -s "$ISING_CACHE" ]]; then
  (
    flock -x 200
    if [[ ! -s "$ISING_CACHE" ]]; then
      echo "[ising] building shared reference -> $ISING_CACHE"
      python -c "
import sys; sys.path.insert(0,'$ROOT_DIR')
from mc_line_core import ising_reference
ising_reference(cache_path='$ISING_CACHE')
"
    fi
  ) 200>"${ISING_CACHE}.lock"
fi

ANCHOR_ARGS=()
[[ "$EPS_ND0" != "-" ]] && ANCHOR_ARGS+=(--eps-nd0 "$EPS_ND0")
[[ "$MU0"     != "-" ]] && ANCHOR_ARGS+=(--mu0     "$MU0")
[[ "$S0"      != "-" ]] && ANCHOR_ARGS+=(--s0      "$S0")
if [[ "$EPS_ND0" == "-" ]]; then
  echo "anchor       : exact lattice-gas point at eps_d=0"
fi

SCAN_ARGS=()
[[ "$SCAN_FALLBACK" == "1" ]] && SCAN_ARGS+=(--scan-fallback)

# No srun.  This is a single-node, single-task allocation whose parallelism is
# internal (multiprocessing over MC replicas across $SLURM_CPUS_PER_TASK
# cores), so srun adds nothing but a job step -- and that job step is what
# fails with "Job credential expired".  Set USE_SRUN=1 to restore it.
set +e
if [[ "${USE_SRUN:-0}" == "1" ]]; then
  LAUNCH=(srun --cpu-bind=none python "$SCRIPT")
else
  LAUNCH=(python "$SCRIPT")
fi
"${LAUNCH[@]}" \
  --system "$SYSTEM" \
  --L "$LSIDE" \
  --seed "$SEED" \
  --eps-d-start "$LO" \
  --eps-d-end "$HI" \
  --eps-d-step "$EPS_D_STEP" \
  "${ANCHOR_ARGS[@]}" \
  "${SCAN_ARGS[@]}" \
  --sweeps-ref "$SWEEPS_REF" \
  --replicas "$REPLICAS" \
  --workers "${SLURM_CPUS_PER_TASK:-8}" \
  --samples-per-replica "$SAMPLES" \
  --burn-frac "$BURN_FRAC" \
  --scout-frac "$SCOUT_FRAC" \
  --init "$INIT" \
  --ising-cache "$ISING_CACHE" \
  --checkpoint "$CKPT" \
  --out "$OUT" \
  $EXTRA_ARGS
RC=$?
set -e

echo "finished     : $(date)  rc=$RC"

# A long segment at L=30 may not fit one walltime slot.  Two cases mean
# "unfinished": the job was killed (no .npz, but a checkpoint exists), or
# python exited 7 (ran clean, did not reach eps_d_end).  Both chain.
if [[ ( ! -s "$OUT" || "$RC" -eq 7 ) && -s "$CKPT" ]]; then
  if (( RESUBMIT_COUNT < MAX_RESUBMIT )); then
    NEXT=$((RESUBMIT_COUNT + 1))
    echo "[chain] incomplete; resubmitting ($NEXT/$MAX_RESUBMIT) from $CKPT"
    sbatch -p "${SLURM_JOB_PARTITION:-volta-cpu}" \
      --array="${SLURM_ARRAY_TASK_ID}" \
      --cpus-per-task="${SLURM_CPUS_PER_TASK:-8}" \
      --time="$(squeue -j "${SLURM_JOB_ID}" -h -o %l 2>/dev/null || echo 1-00:00:00)" \
      -J "${SLURM_JOB_NAME:-mc_line}" \
      --export=ALL,WORKDIR="$WORKDIR",MANIFEST="$MANIFEST",SWEEPS_REF="$SWEEPS_REF",\
REPLICAS="$REPLICAS",SAMPLES="$SAMPLES",SCOUT_FRAC="$SCOUT_FRAC",\
EPS_D_STEP="$EPS_D_STEP",INIT="$INIT",SCAN_FALLBACK="$SCAN_FALLBACK",\
RESUBMIT_COUNT="$NEXT" \
      "${ROOT_DIR}/run_mc_line.sh"
    exit 0
  else
    echo "[chain] resubmit cap reached without finishing -- inspect $CKPT"
    exit 1
  fi
fi

# Report the truth.  `set +e` above is only there to capture RC for the
# chaining branch; without these checks the script would fall through to
# "DONE" on a failed run and SLURM would believe the task succeeded.
if (( RC != 0 )); then
  echo "FAILED: python exited $RC -- see the traceback above"
  exit "$RC"
fi
if [[ ! -s "$OUT" ]]; then
  echo "FAILED: python exited 0 but wrote no output at $OUT"
  exit 3
fi

echo "DONE: $OUT"
