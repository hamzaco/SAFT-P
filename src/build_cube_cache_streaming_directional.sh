#!/bin/bash
#SBATCH -J cube_cache
#SBATCH -o cube_cache.%j.out
#SBATCH -e cube_cache.%j.err
#SBATCH -p volta-cpu
#SBATCH -N 1
#SBATCH -n 10
#SBATCH --ntasks-per-node=10
#SBATCH --time=7-00:00:00
# Request all memory on the allocated node.  There was no --mem line here, so
# the job took the partition default, which is what OOM-killed the 13-species
# builds (base patch 110000 / 111000).  Replace 0 with an explicit size if your
# partition rejects --mem=0.
#SBATCH --mem=0

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

BASE_PATCH="${BASE_PATCH:-1,1,1,0,0,0}"
VACANCY_TYPE="${VACANCY_TYPE:-2}"
# CASE_TAG defaults to the base patch, NOT a fixed string.  It used to default to
# "cube_case" for every geometry, so two runs with different BASE_PATCH shared
# PATCHES_NPY, WORKDIR, KEYS_DIR, EXPAND_DIR, MERGED_KEYS and FINAL_CACHE.  Two
# such jobs running at once destroy each other: CLEAN_BUILD=1 makes whichever
# starts second "rm -rf" the first one's stage outputs mid-run, and the patches
# file is reused rather than regenerated, so the second job silently builds the
# FIRST job's geometry under its own name.  Set CASE_TAG explicitly to override.
CASE_TAG="${CASE_TAG:-cube_${BASE_PATCH//,/}}"
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
# Number of sequential waves for the expand stage.  Every expand task holds its
# whole shard in memory at once, so N tasks on one node hold the entire
# configuration set no matter how finely it is sharded -- more shards do not
# reduce the node total, only more waves do.  1 = all shards at once (previous
# behaviour).  Raise it if the expand stage is still OOM-killed.
EXPAND_WAVES="${EXPAND_WAVES:-1}"

# ---------------------------------------------------------------------------
# Exclusive lock on CASE_TAG.
#
# Every path in this script derives from CASE_TAG, and the CLEAN_BUILD block
# below is destructive.  Two jobs sharing a CASE_TAG -- whether by using the
# default for different base patches, or by passing the same CASE_TAG twice --
# delete and overwrite each other's files mid-run, which shows up as
# FileNotFoundError, half-written .npy files ("This file contains pickled
# (object) data"), or a cache silently built from the wrong geometry.  Take the
# lock first and refuse to start if another live job holds it.
# ---------------------------------------------------------------------------
mkdir -p "$WORKDIR"
LOCKFILE="${WORKDIR}/.owner_job"
if ! ( set -o noclobber; echo "${SLURM_JOB_ID}" > "$LOCKFILE" ) 2>/dev/null; then
  OTHER_JOB="$(cat "$LOCKFILE" 2>/dev/null || true)"
  if [[ -n "$OTHER_JOB" && "$OTHER_JOB" != "$SLURM_JOB_ID" ]] \
     && squeue -h -j "$OTHER_JOB" 2>/dev/null | grep -q .; then
    echo "[lock] ERROR: job $OTHER_JOB is already building CASE_TAG=$CASE_TAG" >&2
    echo "[lock]        in $WORKDIR and is still in the queue." >&2
    echo "[lock]        Every path here derives from CASE_TAG, so two jobs sharing" >&2
    echo "[lock]        one destroy each other.  Give this run its own CASE_TAG:" >&2
    echo "[lock]          CASE_TAG=cube_\${BASE_PATCH//,/} BASE_PATCH=$BASE_PATCH sbatch $0" >&2
    exit 1
  fi
  echo "[lock] taking over stale lock from job ${OTHER_JOB:-<empty>} (no longer queued)"
  echo "${SLURM_JOB_ID}" > "$LOCKFILE"
fi
trap 'rm -f "$LOCKFILE"' EXIT
echo "[lock] holding CASE_TAG=$CASE_TAG (job $SLURM_JOB_ID)"

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
echo "case tag     : $CASE_TAG"
echo "patches file : $PATCHES_NPY"
echo "workdir      : $WORKDIR"
echo "final cache  : $FINAL_CACHE"
echo "store cfg face ids : $STORE_CFG_FACE_IDS"
echo "post chunk rows      : $POST_CHUNK_ROWS"
echo "clean build  : $CLEAN_BUILD"
echo "=================================================="

# Always regenerate (it costs milliseconds) and refuse to continue if an
# existing patches file does not match BASE_PATCH.  Reusing a stale file is
# silent and produces a cache labelled with one geometry but built from another.
echo "[prep] generating patches for base patch $BASE_PATCH"
# Job-unique temp name: a shared "<case>.new.npy" is itself a race between two
# jobs, and that race is what produced the misleading "does not match" errors
# (one job's mv removed the file the other was about to read; the other read a
# half-written file and reported it as pickled object data).
PATCHES_TMP="${PATCHES_NPY%.npy}.new.${SLURM_JOB_ID}.npy"
python "$SCRIPT" build-patches \
  --base-patch "$BASE_PATCH" \
  --vacancy-type "$VACANCY_TYPE" \
  --out "$PATCHES_TMP"
if [[ ! -f "$PATCHES_TMP" ]]; then
  echo "[prep] ERROR: build-patches did not produce $PATCHES_TMP" >&2
  exit 1
fi

if [[ -f "$PATCHES_NPY" ]]; then
  # Three distinct outcomes, reported as three distinct messages:
  #   0 = matches, 1 = genuinely a different geometry, 2 = existing file unreadable.
  python - <<PY
import numpy as np, sys
try:
    a = np.load(r"$PATCHES_NPY", allow_pickle=False)
except Exception as exc:
    print(f"[prep] existing patches file is unreadable: {type(exc).__name__}: {exc}", file=sys.stderr)
    sys.exit(2)
b = np.load(r"$PATCHES_TMP", allow_pickle=False)
sys.exit(0 if (a.shape == b.shape and np.array_equal(a, b)) else 1)
PY
  case $? in
    0) echo "[prep] existing patches match BASE_PATCH" ;;
    1) echo "[prep] ERROR: $PATCHES_NPY holds a different geometry than BASE_PATCH=$BASE_PATCH." >&2
       echo "[prep]        CASE_TAG=$CASE_TAG was used for another base patch." >&2
       echo "[prep]        Use a CASE_TAG unique to this geometry, e.g." >&2
       echo "[prep]          CASE_TAG=cube_\${BASE_PATCH//,/} BASE_PATCH=$BASE_PATCH sbatch $0" >&2
       rm -f "$PATCHES_TMP"; exit 1 ;;
    2) echo "[prep] ERROR: $PATCHES_NPY exists but could not be read (see above)." >&2
       echo "[prep]        It is most likely truncated by a concurrent job, or left over" >&2
       echo "[prep]        from an interrupted run.  Delete it and resubmit." >&2
       rm -f "$PATCHES_TMP"; exit 1 ;;
  esac
fi
mv -f "$PATCHES_TMP" "$PATCHES_NPY"

python - <<PY
import numpy as np
p = np.load(r"$PATCHES_NPY")
print("patches.shape =", p.shape)
print("first rows:")
print(p[:min(5, len(p))])
PY

echo "[preflight] projected size"
python - <<PY
import numpy as np, sys
sys.path.insert(0, r"$ROOT_DIR")
from cube_cache_builder_streaming_directional import _build_corner_lookup, _min_uint_dtype
p = np.load(r"$PATCHES_NPY")
S = int(p.shape[0]); npt = int(p.max()) + 1
_, tv, _ = _build_corner_lookup(p)
flats = int(np.prod([len(v) for v in tv]))
total = S ** 8
cfg_b = np.dtype(_min_uint_dtype(max(S - 1, 0))).itemsize
GB = 1e9
shards = int("${SLURM_NTASKS:-1}"); waves = max(1, int("${EXPAND_WAVES:-1}"))
concurrent = max(1, -(-shards // waves))
print(f"  species                 : {S}")
print(f"  boundary flats          : {flats:,}")
print(f"  configurations (S**8)   : {total:,}")
print(f"  expand: {shards} shards in {waves} wave(s) -> {concurrent} concurrent")
cache = total * (8 * cfg_b + S + npt * npt)
print(f"  expand peak RAM/node    : {total/shards*8*cfg_b*concurrent/GB:8.1f} GB")
print(f"  merge-cache peak RAM    : {'      ~2-3':>8s} GB (cfg/species_counts/bond_hist are memmapped)")
print(f"  DISK needed for cache   : {cache/GB:8.1f} GB  <- must be free under FINAL_CACHE")
import shutil, os
d = os.path.dirname(r"$FINAL_CACHE") or "."
os.makedirs(d, exist_ok=True)
free = shutil.disk_usage(d).free
print(f"  disk free there         : {free/GB:8.1f} GB" + ("" if free > cache * 1.05 else "   *** NOT ENOUGH ***"))
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

echo "[stage 2] expand-shard  (${EXPAND_WAVES} wave(s))"
# The shard count stays $SLURM_NTASKS -- merge-cache checks for exactly that
# many shard files -- but the tasks are launched in EXPAND_WAVES sequential
# groups so only a fraction of them hold a shard in memory at the same time.
PER_WAVE=$(( (SLURM_NTASKS + EXPAND_WAVES - 1) / EXPAND_WAVES ))
for (( wave=0; wave<EXPAND_WAVES; wave++ )); do
  OFFSET=$(( wave * PER_WAVE ))
  NTASK=$(( SLURM_NTASKS - OFFSET ))
  if (( NTASK <= 0 )); then break; fi
  if (( NTASK > PER_WAVE )); then NTASK=$PER_WAVE; fi
  echo "[stage 2] wave $((wave+1))/$EXPAND_WAVES: shards $OFFSET..$((OFFSET+NTASK-1))"
  srun --kill-on-bad-exit=1 --ntasks=$NTASK --cpus-per-task=1 bash -lc '
cd "'"$ROOT_DIR"'"
python "'"$SCRIPT"'" expand-shard \
  --patches "'"$PATCHES_NPY"'" \
  --merged-keys "'"$MERGED_KEYS"'" \
  --out-dir "'"$EXPAND_DIR"'" \
  --shard-id "$(( SLURM_PROCID + '"$OFFSET"' ))" \
  --n-shards "'"$SLURM_NTASKS"'"
'
done

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
