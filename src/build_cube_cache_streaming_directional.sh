#!/bin/bash
#SBATCH -J cube_cache
#SBATCH -o cube_cache.%j.out
#SBATCH -e cube_cache.%j.err
#SBATCH -p volta-cpu
#SBATCH -N 1
#SBATCH -n 10
#SBATCH --ntasks-per-node=10
#SBATCH --time=7-00:00:00

set -euo pipefail

source /fast/shared/anaconda/2025.12/etc/profile.d/conda.sh
conda activate myenv

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export BLIS_NUM_THREADS=1

ROOT_DIR="${SLURM_SUBMIT_DIR}"
SCRIPT="${ROOT_DIR}/cube_cache_builder_streaming_directional.py"

CASE_TAG="${CASE_TAG:-cube_case}"
BASE_PATCH="${BASE_PATCH:-1,1,1,0,0,0}"
VACANCY_TYPE="${VACANCY_TYPE:-2}"
PATCHES_NPY="${PATCHES_NPY:-${ROOT_DIR}/${CASE_TAG}_patches.npy}"
WORKDIR="${WORKDIR:-${ROOT_DIR}/build_${CASE_TAG}}"
KEYS_DIR="${KEYS_DIR:-${WORKDIR}/keys}"
EXPAND_DIR="${EXPAND_DIR:-${WORKDIR}/expand}"
MERGED_KEYS="${MERGED_KEYS:-${WORKDIR}/merged_keys.npz}"
FINAL_CACHE="${FINAL_CACHE:-${ROOT_DIR}/cache/${CASE_TAG}_cache_streaming_directional_dir}"
BOUNDARY_CHUNK_SIZE="${BOUNDARY_CHUNK_SIZE:-200000}"
TARGET_SHARDS="${TARGET_SHARDS:-${SLURM_NTASKS}}"
STORE_CFG_FACE_IDS="${STORE_CFG_FACE_IDS:-0}"
POST_CHUNK_ROWS="${POST_CHUNK_ROWS:-500000}"
CLEAN_BUILD="${CLEAN_BUILD:-1}"

if [[ "$CLEAN_BUILD" == "1" ]]; then
  echo "[clean] removing stale stage outputs"
  rm -rf "$KEYS_DIR" "$EXPAND_DIR"
  rm -f "$MERGED_KEYS"
  rm -rf "$FINAL_CACHE"
fi

mkdir -p "$WORKDIR" "$KEYS_DIR" "$EXPAND_DIR" "$(dirname "$FINAL_CACHE")"
cd "$ROOT_DIR"

echo "=================================================="
echo "job id       : $SLURM_JOB_ID"
echo "partition    : $SLURM_JOB_PARTITION"
echo "nodelist     : $SLURM_JOB_NODELIST"
echo "nodes        : $SLURM_JOB_NUM_NODES"
echo "ntasks       : $SLURM_NTASKS"
echo "submit dir   : $ROOT_DIR"
echo "script       : $SCRIPT"
echo "case tag     : $CASE_TAG"
echo "base patch   : $BASE_PATCH"
echo "patches file : $PATCHES_NPY"
echo "workdir      : $WORKDIR"
echo "final cache  : $FINAL_CACHE"
echo "store cfg face ids : $STORE_CFG_FACE_IDS"
echo "post chunk rows      : $POST_CHUNK_ROWS"
echo "clean build  : $CLEAN_BUILD"
echo "=================================================="

if [[ ! -f "$PATCHES_NPY" ]]; then
  echo "[prep] generating patches"
  python "$SCRIPT" build-patches \
    --base-patch "$BASE_PATCH" \
    --vacancy-type "$VACANCY_TYPE" \
    --out "$PATCHES_NPY"
else
  echo "[prep] using existing patches: $PATCHES_NPY"
fi

python - <<PY
import numpy as np
p = np.load(r"$PATCHES_NPY")
print("patches.shape =", p.shape)
print("first rows:")
print(p[:min(5, len(p))])
PY

echo "[stage 1] keys-shard"
srun --kill-on-bad-exit=1 --ntasks=$SLURM_NTASKS --cpus-per-task=1 bash -lc '
cd "'"$ROOT_DIR"'"
python "'"$SCRIPT"'" keys-shard \
  --patches "'"$PATCHES_NPY"'" \
  --out-dir "'"$KEYS_DIR"'" \
  --shard-id "$SLURM_PROCID" \
  --n-shards "'"$SLURM_NTASKS"'" \
  --target-shards "'"$TARGET_SHARDS"'" \
  --boundary-chunk-size "'"$BOUNDARY_CHUNK_SIZE"'"
'

echo "[stage 1] merge-keys"
python "$SCRIPT" merge-keys \
  --patches "$PATCHES_NPY" \
  --keys-dir "$KEYS_DIR" \
  --merged-keys "$MERGED_KEYS" \
  --n-shards "$SLURM_NTASKS"

echo "[stage 1] inspect merged keys"
python - <<PY
import numpy as np
z = np.load(r"$MERGED_KEYS")
print("group_keys.shape =", z["group_keys"].shape)
print("orbit_sizes.shape =", z["orbit_sizes"].shape)
PY

echo "[stage 2] expand-shard"
srun --kill-on-bad-exit=1 --ntasks=$SLURM_NTASKS --cpus-per-task=1 bash -lc '
cd "'"$ROOT_DIR"'"
python "'"$SCRIPT"'" expand-shard \
  --patches "'"$PATCHES_NPY"'" \
  --merged-keys "'"$MERGED_KEYS"'" \
  --out-dir "'"$EXPAND_DIR"'" \
  --shard-id "$SLURM_PROCID" \
  --n-shards "'"$SLURM_NTASKS"'"
'

echo "[stage 2] merge-cache"
MERGE_EXTRA_ARGS=(--post-chunk-rows "$POST_CHUNK_ROWS")
if [[ "$STORE_CFG_FACE_IDS" == "1" ]]; then
  MERGE_EXTRA_ARGS+=(--store-cfg-face-ids)
fi

python "$SCRIPT" merge-cache \
  --patches "$PATCHES_NPY" \
  --merged-keys "$MERGED_KEYS" \
  --expand-dir "$EXPAND_DIR" \
  --final-cache "$FINAL_CACHE" \
  --n-shards "$SLURM_NTASKS" \
  "${MERGE_EXTRA_ARGS[@]}"

echo "[final] validate cache"
python - <<PY
from cube_cache_builder_streaming_directional import load_cube_cache
cache = load_cube_cache(r"$FINAL_CACHE")
expected = 6 * int(cache["n_patch_types"])**4
n_ft = len(cache["small_face_slots"])
print("cache keys sample =", sorted(cache.keys())[:12])
print("version =", cache.get("version"))
print("face_basis_mode =", cache.get("face_basis_mode"))
print("n_patch_types =", cache["n_patch_types"])
print("n_ft =", n_ft)
print("small_face_dirs.shape =", cache.get("small_face_dirs", None).shape if "small_face_dirs" in cache else None)
print("expected_n_ft =", expected)
print("expected_n_ft cache =", cache.get("expected_n_ft"))
print("n_groups =", cache["n_groups"])
print("n_oriented_groups =", cache.get("n_oriented_groups"))
print("oriented_to_group.shape =", cache.get("oriented_to_group", None).shape if "oriented_to_group" in cache else None)
print("reduced_cfg_count =", cache["reduced_cfg_count"])
print("has_cfg_face_ids =", cache.get("has_cfg_face_ids"))
print("cfg_face_ids.shape =", cache.get("cfg_face_ids", None).shape if "cfg_face_ids" in cache else None)
print("boundary_patch_mode =", cache.get("boundary_patch_mode"))
print("boundary_sparse_entries =", cache.get("boundary_sparse_entries"))
print("patches_per_group[min,max,mean] =", cache.get("patches_per_group_min"), cache.get("patches_per_group_max"), cache.get("patches_per_group_mean"))
print("m_patch_sum[min,max,expected] =", cache.get("m_patch_sum_min"), cache.get("m_patch_sum_max"), cache.get("m_patch_sum_expected"))
print("patch_to_species.shape =", cache["patch_to_species"].shape)
print("m_patch dtype =", cache["m_patch"].dtype)
print("patch_to_small[min,max] =", int(cache["patch_to_small"].min()), int(cache["patch_to_small"].max()))
print("cache path =", cache["cache_path"])
assert n_ft == expected, (n_ft, expected)
assert cache.get("face_basis_mode") == "directional_6xM4_deterministic_opposite_faces"
assert cache.get("boundary_patch_mode") == "orientation_resolved_directional_face_patterns"
assert abs(float(cache.get("m_patch_sum_min")) - 6.0) < 1e-5
assert abs(float(cache.get("m_patch_sum_max")) - 6.0) < 1e-5
PY

echo "DONE: $FINAL_CACHE"
