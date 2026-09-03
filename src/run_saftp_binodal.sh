#!/bin/bash
#SBATCH -J saftp_bino
#SBATCH -o logs/saftp_bino_%A_%a.out
#SBATCH -e logs/saftp_bino_%A_%a.err
# Array task: one temperature per task.  Submitted by submit_saftp_binodal.sh.
set -euo pipefail

ROOT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
WORKDIR="${WORKDIR:?WORKDIR not set}"
MANIFEST="${MANIFEST:?MANIFEST not set}"
N_PHI="${N_PHI:-201}"
PHI_LO="${PHI_LO:-0.002}"
PAIR="${PAIR:-BAEF}"

source /fast/shared/anaconda/2025.12/etc/profile.d/conda.sh
conda activate myenv

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"

T=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$MANIFEST")
echo "task ${SLURM_ARRAY_TASK_ID}: T=${T} n_phi=${N_PHI} phi_lo=${PHI_LO} pair=${PAIR} on $(hostname)"

python "${ROOT_DIR}/saftp_binodal_point.py" \
    --T "$T" --n-phi "$N_PHI" --phi-lo "$PHI_LO" --pair "$PAIR" --outdir "$WORKDIR"
