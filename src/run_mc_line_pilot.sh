#!/bin/bash
#SBATCH -J mc_line_pilot
#SBATCH -o logs/mc_line_pilot.%j.out
#SBATCH -e logs/mc_line_pilot.%j.err
#SBATCH -p volta-cpu
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=06:00:00
#
# RUN THIS BEFORE submit_mc_line.sh, AND READ THE OUTPUT.
#
# Four checks, cheapest first.  Each one can invalidate the whole campaign, and
# each costs minutes instead of the thousands of core-hours the production run
# costs:
#
#   1. throughput on THIS partition, and the wall-time projection that follows
#      from it.  The submit script's cost estimate is only as good as the rate
#      you feed it.
#   2. replica independence.  numba's RNG state is process-global and survives
#      fork(); if the per-worker reseed ever regresses, every replica returns
#      the identical chain and the error bars look perfect while carrying no
#      information.  This check compares replica means directly.
#   3. L=10 reproduction.  A short segment from the exact lattice-gas anchor at
#      eps_d=0 must land on the published line.  If it does not, nothing
#      downstream means anything.
#   4. L=20 and L=30 shakedown: two eps_d steps each, at production chain
#      length, to confirm the memory footprint, the tunnelling diagnostic and
#      the real per-step wall time.

set -euo pipefail

source /fast/shared/anaconda/2025.12/etc/profile.d/conda.sh
conda activate myenv

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export NUMBA_NUM_THREADS=1
export NUMBA_THREADING_LAYER=workqueue

ROOT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
WORKDIR="${WORKDIR:-/pool/hamza/mc_line_pilot}"
SYSTEM="${SYSTEM:-l}"
CPUS="${SLURM_CPUS_PER_TASK:-8}"
ISING_CACHE="${WORKDIR}/ising_ref.npz"
export NUMBA_CACHE_DIR="/tmp/numba_cache_${USER}_${SLURM_JOB_ID:-pilot}"
mkdir -p "$WORKDIR" "${ROOT_DIR}/logs" "$NUMBA_CACHE_DIR"
cd "$ROOT_DIR"

echo "############ 1. throughput and projection ############"
python mc_line_sweep.py --benchmark --system "$SYSTEM" --L 10,20,30

echo
echo "############ 2. replica independence ############"
python - <<'PY'
import os, sys, numpy as np, multiprocessing as mp
sys.path.insert(0, os.getcwd())
from mc_line_core import (SYSTEMS, get_species_list_ind, make_lattices,
                          build_data_mu, _worker_init, lattice_gas_anchor)

sysname = os.environ.get("SYSTEM", "l")
species = get_species_list_ind(SYSTEMS[sysname]["species"])
n = len(species)
idx = np.arange(n-1, dtype=np.int64)
e_nd, mu = lattice_gas_anchor(sysname)
rng = np.random.default_rng(0)
lat = make_lattices(species, 10, "middle", rng)

payload = dict(species=species, species_mu_indices=idx, empty_index=n-1,
               beta=1.0)
pool = mp.get_context("fork").Pool(4, initializer=_worker_init,
                                   initargs=(payload, 1))
d = build_data_mu(pool, lat, species, idx, e_nd, 0.0, mu, 1.0,
                  2_000_000, 2_000, 1200, 1000, 4, 0.1,
                  np.random.default_rng(1), 10, call_id=1)
pool.close(); pool.join()

means = [p["mean_rho"] for p in d.diag["per_replica"]]
print("replica mean rho:", ["%.6f" % m for m in means])
spread = max(means) - min(means)
print("spread = %.3e" % spread)
if spread < 1e-12:
    print("FAIL: replicas are identical -- the numba reseed is not working.")
    print("      Every error bar in this campaign would be meaningless.")
    raise SystemExit(1)
print("PASS: replicas differ, so the per-worker numba reseed is live.")
PY

echo
echo "############ 3. L=10 reproduction from the exact eps_d=0 anchor ############"
echo "# 5 continuation steps at production statistics; eps_nd should track"
echo "# the logged line.  Compare against:"
python mc_line_manifest.py --system "$SYSTEM" --list-anchors --L 10 \
    --eps-d-min 0 --eps-d-max 0.12 --width 0.02 || true
python mc_line_sweep.py \
    --system "$SYSTEM" --L 10 --seed 1 \
    --eps-d-start 0.0 --eps-d-end 0.10 --eps-d-step 0.02 \
    --replicas "$CPUS" --workers "$CPUS" \
    --ising-cache "$ISING_CACHE" \
    --scan-fallback \
    --out "${WORKDIR}/pilot_${SYSTEM}_L10.npz"

echo
echo "############ 4. L=20 and L=30 shakedown (2 steps each) ############"
for L in 20 30; do
  echo "---- L=$L ----"
  ANCH=$(python mc_line_manifest.py --system "$SYSTEM" --L "$L" \
           --eps-d-min 1.80 --eps-d-max 1.84 --width 0.04 --seeds 1 \
           | head -1)
  echo "manifest line: $ANCH"
  read -r _ _ _ LO HI E0 M0 _ SD <<< "$ANCH"
  python mc_line_sweep.py \
      --system "$SYSTEM" --L "$L" --seed "$SD" \
      --eps-d-start "$LO" --eps-d-end "$HI" --eps-d-step 0.02 \
      --eps-nd0 "$E0" --mu0 "$M0" \
      --replicas "$CPUS" --workers "$CPUS" \
      --ising-cache "$ISING_CACHE" \
      --scan-fallback \
      --checkpoint "${WORKDIR}/ck_pilot_${SYSTEM}_L${L}.json" \
      --out "${WORKDIR}/pilot_${SYSTEM}_L${L}.npz"
done

echo
echo "############ what to read off ############"
cat <<'EOF'
 * "steps with a non-tunnelling replica" must be 0.  Any nonzero count at L=30
   means the chain is not ergodic at that length and the located point is not a
   critical point -- raise SWEEPS_REF or lower EPS_D_MAX before committing.
 * "stalled steps" counts L-BFGS-B terminating at its starting point.  A few
   are tolerable (--scan-fallback redoes them); a majority means the eps_d step
   is too small to displace the objective at this L.
 * ESS/n below ~0.1 on any step means the reweighting is being carried by a
   handful of configurations; widen nothing, add statistics.
 * The per-step wall time printed here is what submit_mc_line.sh's projection
   should be reconciled against.  Pass the measured rate as RATE=... .
EOF
echo "DONE. artefacts in $WORKDIR"
