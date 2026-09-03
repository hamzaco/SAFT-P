#!/bin/bash
# submit_mc_line.sh -- build the manifests and submit the finite-size
# critical-LINE campaign (L = 10, 20, 30).
#
#   cd src
#   bash submit_mc_line.sh --dry-run    # ALWAYS do this first: it prints the
#                                       # core-hour cost before you commit
#   bash submit_mc_line.sh
#   EPS_D_MAX=2.5 bash submit_mc_line.sh          # cheaper, covers eps_d=1.96
#   SIZES="10 20" SEEDS="1 2" bash submit_mc_line.sh
#
# One SLURM array per lattice size, so each array carries its own walltime and
# the L=10 tasks are not stuck behind the L=30 request.  Every task uses
# $CPUS cores (= $CPUS MC replicas run in parallel by multiprocessing) and
# covers one segment of the eps_d sweep, anchored on the logged L=10
# continuation at its left edge (see mc_line_manifest.py).
#
# Cost scaling, for planning: the wall time of one eps_d step goes as
# (L/10)^2.17 * L^2, i.e. 1 : 18 : 98 for L = 10 : 20 : 30.  The segment width
# is scaled inversely so tasks stay roughly the same length; the number of
# tasks therefore grows with L.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKDIR="${WORKDIR:-/pool/hamza/mc_line}"
SYSTEMS="${SYSTEMS:-l}"                 # "l stick" for both geometries
SIZES="${SIZES:-10 20 30}"
# Seeds multiply the cost linearly, and at L=30 the cost is already the whole
# budget, so the default is ONE seed.  The noise estimate you actually get for
# free is different and cheaper: neighbouring segments overlap by $OVERLAP, so
# every overlap is an independent repeat of the same eps_d, and the scatter of
# the line about a smooth curve bounds the step-to-step error.  Add seeds at
# L=10 (cheap) if you want a direct check that those two agree.
SEEDS="${SEEDS:-1}"
EPS_D_MIN="${EPS_D_MIN:-0.0}"
EPS_D_MAX="${EPS_D_MAX:-6.0}"
EPS_D_STEP="${EPS_D_STEP:-0.02}"        # notebook value
OVERLAP="${OVERLAP:-0.04}"              # 2 steps of overlap between segments
SWEEPS_REF="${SWEEPS_REF:-6e6}"         # 6e6 == the notebook's 600e6 at L=10
SCOUT_FRAC="${SCOUT_FRAC:-0.08}"
SAMPLES="${SAMPLES:-6000}"
CPUS="${CPUS:-8}"
MEM="${MEM:-}"                          # empty -> partition default
PARTITION="${PARTITION:-volta-cpu}"
TRACE_DIR="${TRACE_DIR:-${ROOT_DIR}/../results_logs}"
FIX_S="${FIX_S:-0}"                     # 1 -> pass the logged s, do not refit
SCAN_FALLBACK="${SCAN_FALLBACK:-1}"
RATE="${RATE:-6e6}"                     # measured attempts/s/core, for costing
DRY=0
[[ "${1:-}" == "--dry-run" ]] && DRY=1

mkdir -p "$WORKDIR" "${ROOT_DIR}/logs"

# ---- per-size segment width and walltime -----------------------------------
# width_for  : how much eps_d one task covers
# walltime_for: SLURM --time, with ~2x headroom over the projection below
width_for() {
  case "$1" in
    10) echo "0.40" ;;
    16) echo "0.20" ;;
    20) echo "0.20" ;;
    24) echo "0.10" ;;
    30) echo "0.10" ;;
    32) echo "0.10" ;;
    *)  echo "0.10" ;;
  esac
}
walltime_for() {
  case "$1" in
    10) echo "04:00:00" ;;
    16) echo "12:00:00" ;;
    20) echo "1-00:00:00" ;;
    24) echo "1-12:00:00" ;;
    30) echo "2-00:00:00" ;;
    32) echo "2-00:00:00" ;;
    *)  echo "2-00:00:00" ;;
  esac
}

echo "workdir    : $WORKDIR"
echo "systems    : $SYSTEMS"
echo "sizes      : $SIZES"
echo "seeds      : $SEEDS"
echo "eps_d      : $EPS_D_MIN .. $EPS_D_MAX  step $EPS_D_STEP  overlap $OVERLAP"
echo "sweeps_ref : $SWEEPS_REF   (steps(L) = sweeps_ref * (L/10)^2.17 * L^2)"
echo "scout_frac : $SCOUT_FRAC"
echo "cpus/task  : $CPUS"
echo "trace dir  : $TRACE_DIR"
echo

TOTAL_CORE_H=0
for SYS in $SYSTEMS; do
for L in $SIZES; do
  W=$(width_for "$L")
  MANIFEST="${WORKDIR}/manifest_${SYS}_L${L}.txt"

  FIX_S_ARG=()
  [[ "$FIX_S" == "1" ]] && FIX_S_ARG+=(--fix-s)

  python "${ROOT_DIR}/mc_line_manifest.py" \
      --system "$SYS" --L "$L" \
      --eps-d-min "$EPS_D_MIN" --eps-d-max "$EPS_D_MAX" \
      --eps-d-step "$EPS_D_STEP" --width "$W" --overlap "$OVERLAP" \
      --seeds "$SEEDS" --trace-dir "$TRACE_DIR" \
      "${FIX_S_ARG[@]}" --out "$MANIFEST"

  N=$(wc -l < "$MANIFEST")
  [[ "$N" -eq 0 ]] && { echo "$SYS L=$L : no tasks, skipping"; continue; }
  T=$(walltime_for "$L")

  # --- cost projection -------------------------------------------------
  COST=$(python - "$L" "$W" "$EPS_D_STEP" "$SWEEPS_REF" "$SCOUT_FRAC" "$N" "$CPUS" "$RATE" <<'PY'
import sys
L, W, step, sref, sfrac, N, cpus, rate = (float(x) for x in sys.argv[1:9])
steps = sref * (L/10.0)**2.17 * L*L
t_chain = steps / rate                      # seconds, one full chain, one core
equiv   = 6.0*sfrac + 1.0                   # scout calls + production, per step
t_step  = t_chain * equiv
n_steps = max(round(W/step), 1)
t_task  = t_step * n_steps
print(f"{t_chain:.6g} {t_step:.6g} {t_task:.6g} {t_task*N*cpus/3600.0:.6g}")
PY
)
  read -r T_CHAIN T_STEP T_TASK CORE_H <<< "$COST"
  TOTAL_CORE_H=$(python -c "print(f'{$TOTAL_CORE_H + $CORE_H:.6g}')")

  printf '%s L=%-3s : %3d tasks x %s cores, width %s, walltime %s\n' \
         "$SYS" "$L" "$N" "$CPUS" "$W" "$T"
  python -c "
def hms(s):
    s=float(s); d=int(s//86400); h=int((s%86400)//3600); m=int((s%3600)//60)
    return (f'{d}d{h:02d}h{m:02d}m' if d else f'{h:d}h{m:02d}m')
print(f'            1 chain {hms($T_CHAIN)}  |  1 eps_d step {hms($T_STEP)}  |  1 task {hms($T_TASK)}  |  {$CORE_H:.0f} core-h')
"

  CMD=(sbatch
       -p "$PARTITION"
       --array="1-${N}"
       --cpus-per-task="$CPUS"
       --time="$T"
       -J "mcline_${SYS}_L${L}")
  [[ -n "$MEM" ]] && CMD+=(--mem="$MEM")
  CMD+=(
       --export=ALL,WORKDIR="$WORKDIR",MANIFEST="$MANIFEST",SWEEPS_REF="$SWEEPS_REF",REPLICAS="$CPUS",SAMPLES="$SAMPLES",SCOUT_FRAC="$SCOUT_FRAC",EPS_D_STEP="$EPS_D_STEP",INIT=middle,SCAN_FALLBACK="$SCAN_FALLBACK"
       "${ROOT_DIR}/run_mc_line.sh")

  if [[ $DRY -eq 1 ]]; then
    printf '            would run:'; printf ' %q' "${CMD[@]}"; echo
  else
    "${CMD[@]}"
  fi
done
done

echo
printf 'TOTAL projected cost: %s core-hours\n' \
       "$(python -c "print(f'{$TOTAL_CORE_H:.0f}')")"
echo "  (at ${RATE} attempts/s/core -- run"
echo "   'python mc_line_sweep.py --benchmark --L 10,20,30 --system l'"
echo "   on this partition first and pass the measured rate as RATE=...)"
python -c "
ch = $TOTAL_CORE_H
for cores in (100, 200, 400):
    print(f'  {ch/cores:8.1f} h wall if {cores} cores are free the whole time')
if ch > 4000:
    print()
    print('  This is a large request.  The two knobs that cut it hardest,')
    print('  in order of how little they cost you scientifically:')
    print('    EPS_D_MAX=2.5   restrict the sweep to the region the paper')
    print('                    actually argues about (it contains eps_d=1.96)')
    print('    SIZES=\"10 20\"   drop L=30; two sizes still extrapolate, but')
    print('                    with zero degrees of freedom, so the residual')
    print('                    stops being a check on the L^(-1/nu) form')
    print('    EPS_D_STEP=0.05 coarser continuation. Cheapest per unit of')
    print('                    eps_d covered, and a larger step actually helps')
    print('                    the L-BFGS-B estimator move -- but it is a')
    print('                    change to the published estimator, so say so.')
"
echo
echo "When the arrays finish:"
echo "  python ${ROOT_DIR}/mc_line_merge.py --workdir $WORKDIR --outdir ${ROOT_DIR}/../figures"
