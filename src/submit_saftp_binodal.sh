#!/bin/bash
# submit_saftp_binodal.sh -- re-run the SAFT-P isomer binodal on a converged
# composition grid, one SLURM task per temperature.
#
#   cd src
#   bash submit_saftp_binodal.sh                 # submit
#   bash submit_saftp_binodal.sh --dry-run
#   TEMPS="0.5 0.6" N_PHI=401 bash submit_saftp_binodal.sh
#   PAIR=AFEB bash submit_saftp_binodal.sh       # enantiomer control
#
# Why: the published Fig. 7 used phi_lo=0.01 with n_phi=51.  For T <= 0.8 the
# common-tangent construction returned exactly phi=0.01 / 0.49 -- the grid
# endpoints -- so those branches are artifacts, not converged coexistence
# compositions.  Cost is one constrained optimisation per grid point and grows
# sharply as T falls, hence one task per temperature.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKDIR="${WORKDIR:-/pool/hamza/saftp_binodal}"
TEMPS="${TEMPS:-0.5 0.6 0.7 0.8 0.9 1.0 1.1}"
N_PHI="${N_PHI:-201}"
PHI_LO="${PHI_LO:-0.002}"
PAIR="${PAIR:-BAEF}"
CPUS="${CPUS:-2}"
PARTITION="${PARTITION:-volta-cpu}"
DRY=0
[[ "${1:-}" == "--dry-run" ]] && DRY=1

mkdir -p "$WORKDIR" "${ROOT_DIR}/logs"

# Walltime is set by temperature: the association solve stiffens as T falls.
# Measured ~20 s per grid point at T=0.5 on one core, ~7 s at T=0.9.
walltime_for() {
  awk -v t="$1" 'BEGIN{ if (t<=0.6) print "16:00:00"; else if (t<=0.8) print "08:00:00"; else print "04:00:00" }'
}

# One array per walltime tier so the warm temperatures are not stuck behind T=0.5.
for TIER in cold warm hot; do
  MANIFEST="${WORKDIR}/manifest_${TIER}.txt"
  : > "$MANIFEST"
  for T in $TEMPS; do
    W=$(walltime_for "$T")
    case "$TIER:$W" in
      cold:16:00:00|warm:08:00:00|hot:04:00:00) echo "$T" >> "$MANIFEST" ;;
    esac
  done
  N=$(wc -l < "$MANIFEST")
  [[ "$N" -eq 0 ]] && { rm -f "$MANIFEST"; continue; }
  case "$TIER" in cold) W="16:00:00";; warm) W="08:00:00";; *) W="04:00:00";; esac

  echo "$TIER : $N task(s), walltime $W  [$(tr '\n' ' ' < "$MANIFEST")]"
  CMD=(sbatch -p "$PARTITION" --array="1-${N}" --cpus-per-task="$CPUS"
       --time="$W" -J "saftp_${TIER}"
       --export=ALL,WORKDIR="$WORKDIR",MANIFEST="$MANIFEST",N_PHI="$N_PHI",PHI_LO="$PHI_LO",PAIR="$PAIR"
       "${ROOT_DIR}/run_saftp_binodal.sh")
  if [[ $DRY -eq 1 ]]; then printf '  would run: %q ' "${CMD[@]}"; echo; else "${CMD[@]}"; fi
done

echo
echo "workdir : $WORKDIR   grid : n_phi=$N_PHI, phi_lo=$PHI_LO  (Dx resolution $(awk -v n="$N_PHI" -v p="$PHI_LO" 'BEGIN{printf "%.4f", 4*(0.5-2*p)/(n-1)}'))"
echo "collect : python ${ROOT_DIR}/saftp_binodal_collect.py --workdir $WORKDIR"
