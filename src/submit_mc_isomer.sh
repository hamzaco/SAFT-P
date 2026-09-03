#!/bin/bash
# submit_mc_isomer.sh -- revised Fig. 7 isomer Monte Carlo campaign.
#
# Correct interpretation of the three modes:
#   wl          authoritative finite-L equilibrium binodal; reweights F(Delta x)
#               to the equal-basin-weight coexistence field mu_coex.
#   branch      fixed-field seeded diagnostic.  Useful at mu1=0 to show that
#               asymmetry is not created solely by an applied field.
#   branch-coex seeded validation at the WL-derived mu_coex.  These jobs must
#               run only after matching WL jobs have completed.
#
# Examples
# --------
# Dry run:
#   bash submit_mc_isomer.sh --dry-run
#
# Full small-size campaign:
#   MODES="branch wl branch-coex" SIZES="20 30" bash submit_mc_isomer.sh
#
# Add the sizes needed for a thermodynamic-limit WL extrapolation while reusing
# existing L=20,30 results already in WORKDIR:
#   MODES="wl branch-coex" SIZES="40 50 60" bash submit_mc_isomer.sh
#
# Explicitly test whether T*=0.55 (T=1.10) still has two-phase WL structure:
#   MODES=wl SIZES="20 30 40 50 60" TEMPS=1.1 bash submit_mc_isomer.sh
#
# Collect/plot afterwards:
#   python mc_isomer_collect_coex.py --workdir "$WORKDIR" \
#       --csv-prefix "$WORKDIR/fig7" --plot-prefix "$WORKDIR/fig7"
set -euo pipefail

ROOT_DIR="$pwd"
WORKDIR="${WORKDIR:-/pool/hamza/mc_isomer}"
MODES="${MODES:-branch wl branch-coex}"
TEMPS="${TEMPS:-0.7 0.8 0.9 1.0}"
SIZES="${SIZES:-20 30}"
FIELDS="${FIELDS:-0.0 0.042}"       # fixed-field branch diagnostics only
WL_MU1="${WL_MU1:-0.042}"           # WL sampling tilt only; NOT assumed coexistence
BRANCH_SEEDS="${BRANCH_SEEDS:-1 2 3 4}"
COEX_SEEDS="${COEX_SEEDS:-1 2 3 4}"
WL_SEEDS="${WL_SEEDS:-1 2 3}"
SWEEPS="${SWEEPS:-500000}"          # seeded branch modes: attempted moves/site
PARTITION="${PARTITION:-volta-cpu}"
DRY=0
[[ "${1:-}" == "--dry-run" ]] && DRY=1

mkdir -p "$WORKDIR" "${ROOT_DIR}/logs"

declare -A WL_JOB_ID

has_mode() {
  [[ " $MODES " == *" $1 "* ]]
}

walltime_for() {   # mode, L
  case "$1:$2" in
    branch:20|branch-coex:20) echo "01:00:00" ;;
    branch:*|branch-coex:*)   echo "04:00:00" ;;
    wl:20)                    echo "04:00:00" ;;
    wl:30)                    echo "16:00:00" ;;
    wl:*)                     echo "24:00:00" ;;
  esac
}

make_manifest() { # mode, L, manifest
  local mode="$1" L="$2" manifest="$3"
  : > "$manifest"
  for T in $TEMPS; do
    case "$mode" in
      branch)
        for MU in $FIELDS; do
          for S in $BRANCH_SEEDS; do
            echo "$mode $T $L $MU $S" >> "$manifest"
          done
        done
        ;;
      wl)
        for S in $WL_SEEDS; do
          echo "$mode $T $L $WL_MU1 $S" >> "$manifest"
        done
        ;;
      branch-coex)
        for S in $COEX_SEEDS; do
          # MU1 is a manifest placeholder. run_mc_isomer.sh obtains mu_coex from WL_SOURCE.
          echo "$mode $T $L 0.0 $S" >> "$manifest"
        done
        ;;
      *)
        echo "Unknown mode in make_manifest: $mode" >&2
        exit 2
        ;;
    esac
  done
}

submit_one() { # mode, L, optional dependency jobid
  local mode="$1" L="$2" dep="${3:-}"
  local manifest="${WORKDIR}/manifest_${mode}_L${L}.txt"
  make_manifest "$mode" "$L" "$manifest"
  local N
  N=$(wc -l < "$manifest")
  [[ "$N" -eq 0 ]] && return 0
  local W
  W=$(walltime_for "$mode" "$L")

  local CMD=(sbatch --parsable -p "$PARTITION" --array="1-${N}" --cpus-per-task=1 --time="$W"
             -J "iso_${mode}_L${L}"
             --export=ALL,WORKDIR="$WORKDIR",WL_SOURCE="$WORKDIR",MANIFEST="$manifest",SWEEPS="$SWEEPS")
  if [[ -n "$dep" ]]; then
    CMD+=(--dependency="afterok:${dep}")
  fi
  CMD+=("${ROOT_DIR}/run_mc_isomer.sh")

  echo "$mode L=$L : $N tasks, walltime $W${dep:+, afterok $dep}" >&2
  if [[ $DRY -eq 1 ]]; then
    printf '  would run:' >&2
    printf ' %q' "${CMD[@]}" >&2
    echo >&2
    echo "DRY_${mode}_${L}"
  else
    local jid
    jid=$("${CMD[@]}")
    # --parsable can return jobid or jobid;cluster
    jid="${jid%%;*}"
    echo "$jid"
  fi
}

# 1) Fixed-field branch diagnostics are independent and can be launched first.
if has_mode branch; then
  for L in $SIZES; do
    submit_one branch "$L" >/dev/null
  done
fi

# 2) WL must precede branch-coex when both are requested in this invocation.
if has_mode wl; then
  for L in $SIZES; do
    WL_JOB_ID[$L]="$(submit_one wl "$L")"
  done
fi

# 3) Coexistence-field seeded validation.  If WL was submitted above, use an
# afterok dependency.  If MODES excludes wl, existing valid WL JSONs in WORKDIR
# are assumed and branch-coex starts immediately.
if has_mode branch-coex; then
  for L in $SIZES; do
    dep=""
    if [[ -n "${WL_JOB_ID[$L]:-}" && "${WL_JOB_ID[$L]}" != DRY_* ]]; then
      dep="${WL_JOB_ID[$L]}"
    fi
    submit_one branch-coex "$L" "$dep" >/dev/null
  done
fi

echo
echo "workdir : $WORKDIR"
echo "collect : python ${ROOT_DIR}/mc_isomer_collect_coex.py --workdir $WORKDIR --csv-prefix $WORKDIR/fig7 --plot-prefix $WORKDIR/fig7"
