from __future__ import annotations

import argparse
import gzip
import hashlib
import pickle
from itertools import permutations, product
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np

N, E, Sdir, W, TOP, BOT = 0, 1, 2, 3, 4, 5

# Corner order:
#   0:000, 1:010, 2:100, 3:110, 4:001, 5:011, 6:101, 7:111
CORNER_COORDS = np.array([
    [0, 0, 0],
    [0, 1, 0],
    [1, 0, 0],
    [1, 1, 0],
    [0, 0, 1],
    [0, 1, 1],
    [1, 0, 1],
    [1, 1, 1],
], dtype=np.int64)
COORD_TO_CORNER = {tuple(c.tolist()): i for i, c in enumerate(CORNER_COORDS)}

# Face order == direction order
#   0:N, 1:E, 2:S, 3:W, 4:TOP, 5:BOT
FACE_SLOT_CORNERS = np.array([
    [0, 1, 4, 5],
    [1, 3, 5, 7],
    [2, 3, 6, 7],
    [0, 2, 4, 6],
    [4, 5, 6, 7],
    [0, 1, 2, 3],
], dtype=np.int64)
OPPOSITE_FACE = np.array([Sdir, W, N, E, BOT, TOP], dtype=np.int64)

CORNER_OUTWARD_DIRS = (
    (N,    W,   BOT),
    (N,    E,   BOT),
    (Sdir, W,   BOT),
    (Sdir, E,   BOT),
    (N,    W,   TOP),
    (N,    E,   TOP),
    (Sdir, W,   TOP),
    (Sdir, E,   TOP),
)

CUBE_BONDS = (
    (0, E,    1, W),
    (2, E,    3, W),
    (4, E,    5, W),
    (6, E,    7, W),
    (0, Sdir, 2, N),
    (1, Sdir, 3, N),
    (4, Sdir, 6, N),
    (5, Sdir, 7, N),
    (0, TOP,  4, BOT),
    (1, TOP,  5, BOT),
    (2, TOP,  6, BOT),
    (3, TOP,  7, BOT),
)

DIR_VECS = np.array([
    [-1,  0,  0],
    [ 0,  1,  0],
    [ 1,  0,  0],
    [ 0, -1,  0],
    [ 0,  0,  1],
    [ 0,  0, -1],
], dtype=np.int64)
VEC_TO_DIR = {tuple(v.tolist()): i for i, v in enumerate(DIR_VECS)}


def _build_corner_outward_positions() -> tuple[tuple[tuple[int, int], ...], ...]:
    out = []
    for corner in range(8):
        positions = []
        for d in CORNER_OUTWARD_DIRS[corner]:
            hit = np.where(FACE_SLOT_CORNERS[d] == corner)[0]
            if hit.size != 1:
                raise RuntimeError(f"Could not locate corner {corner} on face {d}.")
            positions.append((d, int(hit[0])))
        out.append(tuple(positions))
    return tuple(out)


def _build_cube_rotation_actions() -> tuple[dict[str, np.ndarray], ...]:
    actions = []
    seen = set()
    for perm in permutations((0, 1, 2)):
        for signs in product((-1, 1), repeat=3):
            R = np.zeros((3, 3), dtype=np.int64)
            for row, col in enumerate(perm):
                R[row, col] = signs[row]
            if int(round(np.linalg.det(R))) != 1:
                continue

            key = tuple(R.ravel().tolist())
            if key in seen:
                continue
            seen.add(key)

            corner_perm = np.empty(8, dtype=np.int64)
            for old_i, c in enumerate(CORNER_COORDS):
                centered = 2 * c - 1
                centered_new = R @ centered
                new_c = ((centered_new + 1) // 2).astype(np.int64)
                corner_perm[old_i] = COORD_TO_CORNER[tuple(new_c.tolist())]

            dir_perm = np.empty(6, dtype=np.int64)
            for old_d, v in enumerate(DIR_VECS):
                dir_perm[old_d] = VEC_TO_DIR[tuple((R @ v).tolist())]

            face_perm = dir_perm.copy()
            slot_perm = np.empty((6, 4), dtype=np.int64)
            for old_face in range(6):
                new_face = face_perm[old_face]
                for old_slot, old_corner in enumerate(FACE_SLOT_CORNERS[old_face]):
                    new_corner = corner_perm[old_corner]
                    hit = np.where(FACE_SLOT_CORNERS[new_face] == new_corner)[0]
                    if hit.size != 1:
                        raise RuntimeError("Failed to build cube face-slot rotation map.")
                    slot_perm[old_face, old_slot] = int(hit[0])

            actions.append({
                "corner_perm": corner_perm,
                "dir_perm": dir_perm,
                "face_perm": face_perm,
                "slot_perm": slot_perm,
            })

    if len(actions) != 24:
        raise RuntimeError(f"Expected 24 proper cube rotations, found {len(actions)}.")
    return tuple(actions)


CORNER_OUTWARD_POSITIONS = _build_corner_outward_positions()
ROT_ACTIONS = _build_cube_rotation_actions()


def _build_rotation_permutations_flat() -> np.ndarray:
    perms = []
    for act in ROT_ACTIONS:
        perm = np.empty(24, dtype=np.int64)
        for old_face in range(6):
            new_face = int(act["face_perm"][old_face])
            for old_slot in range(4):
                new_slot = int(act["slot_perm"][old_face, old_slot])
                old_flat = 4 * old_face + old_slot
                new_flat = 4 * new_face + new_slot
                perm[new_flat] = old_flat
        perms.append(perm)
    return np.asarray(perms, dtype=np.int64)


ROT_FLAT_PERMS = _build_rotation_permutations_flat()
CORNER_FACE_SLOT_IDX = np.array(
    [[4 * face + slot for face, slot in CORNER_OUTWARD_POSITIONS[c]] for c in range(8)],
    dtype=np.int64,
)
FACE_FLAT_IDX = np.array([[4 * f + s for s in range(4)] for f in range(6)], dtype=np.int64)


def parse_base_patch(spec: str) -> np.ndarray:
    vals = [int(x.strip()) for x in spec.split(",") if x.strip() != ""]
    arr = np.asarray(vals, dtype=np.int64)
    if arr.shape != (6,):
        raise ValueError(f"base patch must contain exactly 6 integers, got {spec!r}")
    return arr


def generate_unique_cube_species_rotations(base_patch: np.ndarray) -> np.ndarray:
    base_patch = np.asarray(base_patch, dtype=np.int64)
    if base_patch.shape != (6,):
        raise ValueError(f"base_patch must have shape (6,), got {base_patch.shape}")

    out = []
    seen = set()
    for act in ROT_ACTIONS:
        rotated = np.empty(6, dtype=np.int64)
        rotated[act["dir_perm"]] = base_patch
        key = tuple(int(x) for x in rotated.tolist())
        if key not in seen:
            seen.add(key)
            out.append(np.array(key, dtype=np.int64))
    return np.asarray(sorted(tuple(x.tolist()) for x in out), dtype=np.int64)


def build_patches_with_vacancy(base_patch: np.ndarray, vacancy_type: Optional[int] = None) -> np.ndarray:
    base_patch = np.asarray(base_patch, dtype=np.int64)
    if vacancy_type is None:
        vacancy_type = int(base_patch.max()) + 1
    active = generate_unique_cube_species_rotations(base_patch)
    vacancy = np.full((1, 6), int(vacancy_type), dtype=np.int64)
    return np.vstack([active, vacancy])


def save_patches_npy(path: str | Path, patches: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, np.asarray(patches, dtype=np.int64))


def load_patches_npy(path: str | Path) -> np.ndarray:
    arr = np.load(path)
    arr = np.asarray(arr, dtype=np.int64)
    if arr.ndim != 2 or arr.shape[1] != 6:
        raise ValueError(f"patches must have shape (n_species, 6), got {arr.shape}")
    return arr


def _encode_face_slots(face_slots_row: np.ndarray, n_patch_types: int) -> int:
    a, b, c, d = map(int, face_slots_row.tolist())
    return ((a * n_patch_types + b) * n_patch_types + c) * n_patch_types + d


def _decode_face_id(face_id: int, n_patch_types: int) -> np.ndarray:
    x = int(face_id)
    out = np.empty(4, dtype=np.int64)
    out[3] = x % n_patch_types
    x //= n_patch_types
    out[2] = x % n_patch_types
    x //= n_patch_types
    out[1] = x % n_patch_types
    x //= n_patch_types
    out[0] = x
    return out



def _build_directional_face_basis(n_patch_types: int):
    """Return the full direction-resolved 4-slot face-pattern basis.

    A small association type is

        small_id = face_dir * M**4 + face_pattern_id

    where face_dir is one of (N,E,S,W,TOP,BOT) and face_pattern_id encodes the
    ordered 2x2 face slots in the fixed global-coordinate convention used by
    FACE_SLOT_CORNERS.  Opposite faces use the same transverse-coordinate slot
    order, so face-face contact has a unique identity slot pairing.  No registry
    averaging is used in the scanner.
    """
    n_patch_types = int(n_patch_types)
    face_radix = int(n_patch_types ** 4)
    pattern_ids = np.arange(face_radix, dtype=np.int64)
    pattern_slots = np.vstack([
        _decode_face_id(int(fid), n_patch_types) for fid in pattern_ids
    ]).astype(_min_uint_dtype(max(n_patch_types - 1, 0)), copy=False)

    n_ft = 6 * face_radix
    small_face_ids = np.arange(n_ft, dtype=np.int64)
    small_face_dirs = np.repeat(np.arange(6, dtype=np.int64), face_radix)
    small_face_slots = np.tile(pattern_slots, (6, 1))
    return face_radix, small_face_ids, small_face_dirs.astype(np.int64), small_face_slots


def _build_undirected_face_basis(n_patch_types: int):
    # Backward-compatible alias: this implementation is direction-resolved.
    face_radix, small_face_ids, _small_face_dirs, small_face_slots = _build_directional_face_basis(n_patch_types)
    return face_radix, small_face_ids, small_face_slots


def _cfg_face_ids_from_cfg(patches: np.ndarray, cfg: np.ndarray, n_patch_types: int) -> np.ndarray:
    """Undirected 4-slot face-pattern id for every stored microstate and face.

    cfg_face_ids[row, face_dir] is an integer in [0, M**4). It intentionally
    does not include face_dir. This is for optional debugging/future analysis;
    the scanner may use either orbit-averaged class-level arrays or explicit oriented-state patch arrays.
    """
    patches = np.asarray(patches, dtype=np.int64)
    cfg = np.asarray(cfg, dtype=np.int64)
    face_radix = int(n_patch_types ** 4)
    out_dtype = _min_uint_dtype(face_radix - 1)
    out = np.empty((cfg.shape[0], 6), dtype=out_dtype)
    for face_dir in range(6):
        corners = FACE_SLOT_CORNERS[face_dir]
        slots = np.empty((cfg.shape[0], 4), dtype=np.int64)
        for slot in range(4):
            slots[:, slot] = patches[cfg[:, int(corners[slot])], face_dir]
        face_id = ((slots[:, 0] * n_patch_types + slots[:, 1]) * n_patch_types + slots[:, 2]) * n_patch_types + slots[:, 3]
        out[:, face_dir] = face_id.astype(out_dtype, copy=False)
    return out



def _orbit_averaged_boundary_sparse_from_key(key: np.ndarray, n_patch_types: int):
    """Orbit-averaged undirected face-pattern counts for one canonical cube class.

    A canonical cube class represents a rotational orbit of boundary flats. The
    association layer must not use only the lexicographic representative and it
    must not attach a lattice direction to a face type after cube orientations
    have been quotient/canonicalized. This routine averages the six face
    patterns over all unique rotated boundary flats in the orbit:

        m_{g,t} = (1/|O_g|) sum_{R in O_g} n_t(R g)

    where t is now only a 4-slot face pattern in [0, M**4). The returned weights
    always sum to 6 for every nonempty class.
    """
    key = np.asarray(key, dtype=np.int64)
    n_patch_types = int(n_patch_types)
    face_radix = int(n_patch_types ** 4)
    counts = {}
    orbit_flats = _unique_orbit_flats_from_key(key, n_patch_types)
    if not orbit_flats:
        return np.empty(0, dtype=_min_uint_dtype(face_radix - 1)), np.empty(0, dtype=np.float32)

    for flat in orbit_flats:
        flat = np.asarray(flat, dtype=np.int64)
        for face_dir in range(6):
            face_id = _encode_face_slots(flat[FACE_FLAT_IDX[face_dir]], n_patch_types)
            counts[int(face_id)] = counts.get(int(face_id), 0) + 1

    denom = float(len(orbit_flats))
    face_ids = np.asarray(sorted(counts.keys()), dtype=_min_uint_dtype(face_radix - 1))
    weights = np.asarray([counts[int(t)] / denom for t in face_ids.tolist()], dtype=np.float32)
    return face_ids, weights


def _build_orbit_averaged_patch_arrays(
    group_keys_arr: np.ndarray,
    n_patch_types: int,
    *,
    group_index_dtype=None,
    verbose: bool = True,
    label: str = "cube-cache",
):
    """Build sparse patch arrays using rotation-orbit averaged face-pattern counts.

    Returns arrays with the same contract as the old patch arrays:
      patch_to_species[a] = cube class g
      patch_to_small[a]   = undirected 4-slot face-pattern type t in [0, M**4)
      m_patch[a]          = average multiplicity/count of type t on class g

    Each class can contribute more than six sparse entries because rotating a
    canonical class can expose multiple face-pattern types. The sum of m_patch
    over entries of each class is exactly 6.
    """
    group_keys_arr = np.asarray(group_keys_arr, dtype=np.int64)
    n_groups = int(group_keys_arr.shape[0])
    n_patch_types = int(n_patch_types)
    face_radix = int(n_patch_types ** 4)
    n_small_faces = 6 * face_radix
    small_dtype = _min_uint_dtype(n_small_faces - 1)
    if group_index_dtype is None:
        group_index_dtype = _min_uint_dtype(max(n_groups - 1, 0))

    sizes = np.empty(n_groups, dtype=np.int64)
    total = 0
    min_sum = np.inf
    max_sum = -np.inf
    report_every = max(1, n_groups // 10)

    for g in range(n_groups):
        face_ids, weights = _orbit_averaged_boundary_sparse_from_key(group_keys_arr[g], n_patch_types)
        sizes[g] = int(face_ids.size)
        total += int(face_ids.size)
        s = float(np.sum(weights, dtype=np.float64))
        min_sum = min(min_sum, s)
        max_sum = max(max_sum, s)
        if verbose and ((g + 1) % report_every == 0 or g + 1 == n_groups):
            print(f"[{label}] boundary-orbit-undirected pass1 {g + 1}/{n_groups}; sparse_entries={total}", flush=True)

    patch_group_ptr = np.empty(n_groups + 1, dtype=np.int64)
    patch_group_ptr[0] = 0
    np.cumsum(sizes, out=patch_group_ptr[1:])
    total_entries = int(patch_group_ptr[-1])

    patch_to_species = np.empty(total_entries, dtype=group_index_dtype)
    patch_to_small = np.empty(total_entries, dtype=small_dtype)
    # float32 is enough for fractions like k/24 and saves substantial cache/RSS.
    # The scanner promotes arithmetic to float64 where needed.
    m_patch = np.empty(total_entries, dtype=np.float32)

    report_every = max(1, n_groups // 10)
    for g in range(n_groups):
        lo = int(patch_group_ptr[g])
        hi = int(patch_group_ptr[g + 1])
        face_ids, weights = _orbit_averaged_boundary_sparse_from_key(group_keys_arr[g], n_patch_types)
        if hi - lo != face_ids.size:
            raise RuntimeError(f"Boundary sparse-size mismatch for group {g}: pass1={hi-lo}, pass2={face_ids.size}")
        patch_to_species[lo:hi] = g
        patch_to_small[lo:hi] = face_ids.astype(small_dtype, copy=False)
        m_patch[lo:hi] = weights.astype(np.float32, copy=False)
        if verbose and ((g + 1) % report_every == 0 or g + 1 == n_groups):
            print(f"[{label}] boundary-orbit-undirected pass2 {g + 1}/{n_groups}", flush=True)

    mean_size = float(np.mean(sizes)) if n_groups else 0.0
    max_size = int(np.max(sizes)) if n_groups else 0
    min_size = int(np.min(sizes)) if n_groups else 0
    stats = {
        "boundary_patch_mode": "rotation_orbit_averaged_undirected_face_patterns",
        "boundary_sparse_entries": int(total_entries),
        "patches_per_group_min": min_size,
        "patches_per_group_max": max_size,
        "patches_per_group_mean": mean_size,
        "m_patch_sum_min": float(min_sum),
        "m_patch_sum_max": float(max_sum),
        "m_patch_sum_expected": 6.0,
    }
    if verbose:
        print(
            f"[{label}] boundary_patch_mode={stats['boundary_patch_mode']} "
            f"entries={total_entries} per_group[min,max,mean]=({min_size},{max_size},{mean_size:.3f}) "
            f"m_sum[min,max]=({stats['m_patch_sum_min']:.6g},{stats['m_patch_sum_max']:.6g})",
            flush=True,
        )
    return patch_to_species, patch_to_small, m_patch, patch_group_ptr, stats



def _boundary_count_sparse_from_flat(flat: np.ndarray, n_patch_types: int):
    """Integer direction-resolved face-pattern list for one oriented cube boundary flat.

    For an oriented cube state, the six exposed faces have definite lab-frame
    directions.  The small association type is

        small_id = face_dir * M**4 + ordered_face_pattern_id.

    Therefore every oriented cube state contributes exactly six sparse entries,
    each with m_patch=1.  No orbit averaging and no registry averaging are hidden
    in m_patch.
    """
    flat = np.asarray(flat, dtype=np.int64)
    face_radix = int(n_patch_types ** 4)
    n_ft = 6 * face_radix
    small_dtype = _min_uint_dtype(n_ft - 1)
    small_ids = np.empty(6, dtype=np.int64)
    for face_dir in range(6):
        fid = _encode_face_slots(flat[FACE_FLAT_IDX[face_dir]], n_patch_types)
        small_ids[face_dir] = int(face_dir) * face_radix + int(fid)
    # Direction is part of the id, so the six entries are normally unique.  Use
    # unique anyway to keep the same sparse contract if conventions change.
    uniq, cnt = np.unique(small_ids, return_counts=True)
    return uniq.astype(small_dtype, copy=False), cnt.astype(np.float32, copy=False), small_ids.astype(small_dtype, copy=False)


def _build_orientation_resolved_patch_arrays(
    group_keys_arr: np.ndarray,
    n_patch_types: int,
    *,
    group_index_dtype=None,
    oriented_index_dtype=None,
    verbose: bool = True,
    label: str = "cube-cache",
):
    """Build sparse association arrays for explicit oriented cube states.

    Canonical groups g still define the internal partition function.  Association
    states are expanded to alpha=(g,R), one for each unique rotated boundary flat.

      oriented_to_group[alpha] = g
      patch_to_species[a]      = alpha
      patch_to_small[a]        = ordered 4-slot face-pattern t
      m_patch[a]               = integer count of t on that oriented boundary

    This avoids the old annealed orientation average
        (1/|O_g|) sum_R n_t(Rg)
    and lets the variational solver choose orientation populations through
    rho_alpha.
    """
    group_keys_arr = np.asarray(group_keys_arr, dtype=np.int64)
    n_groups = int(group_keys_arr.shape[0])
    n_patch_types = int(n_patch_types)
    face_radix = int(n_patch_types ** 4)
    n_small_faces = 6 * face_radix
    small_dtype = _min_uint_dtype(n_small_faces - 1)
    if group_index_dtype is None:
        group_index_dtype = _min_uint_dtype(max(n_groups - 1, 0))

    orientation_counts = np.empty(n_groups, dtype=np.int64)
    sparse_sizes = []
    total_entries = 0
    total_oriented = 0
    min_sum = np.inf
    max_sum = -np.inf
    report_every = max(1, n_groups // 10)

    for g in range(n_groups):
        flats = _unique_orbit_flats_from_key(group_keys_arr[g], n_patch_types)
        orientation_counts[g] = len(flats)
        for flat in flats:
            _, cnt, _ = _boundary_count_sparse_from_flat(flat, n_patch_types)
            sz = int(cnt.size)
            sparse_sizes.append(sz)
            total_entries += sz
            s = float(np.sum(cnt, dtype=np.float64))
            min_sum = min(min_sum, s)
            max_sum = max(max_sum, s)
        total_oriented += int(orientation_counts[g])
        if verbose and ((g + 1) % report_every == 0 or g + 1 == n_groups):
            print(
                f"[{label}] boundary-oriented pass1 {g + 1}/{n_groups}; "
                f"oriented_states={total_oriented} sparse_entries={total_entries}",
                flush=True,
            )

    if oriented_index_dtype is None:
        oriented_index_dtype = _min_uint_dtype(max(total_oriented - 1, 0))

    orientation_group_ptr = np.empty(n_groups + 1, dtype=np.int64)
    orientation_group_ptr[0] = 0
    np.cumsum(orientation_counts, out=orientation_group_ptr[1:])

    patch_group_ptr = np.empty(total_oriented + 1, dtype=np.int64)
    patch_group_ptr[0] = 0
    np.cumsum(np.asarray(sparse_sizes, dtype=np.int64), out=patch_group_ptr[1:])

    patch_to_species = np.empty(total_entries, dtype=oriented_index_dtype)
    patch_to_small = np.empty(total_entries, dtype=small_dtype)
    m_patch = np.empty(total_entries, dtype=np.float32)
    oriented_to_group = np.empty(total_oriented, dtype=group_index_dtype)
    orientation_face_keys = np.empty((total_oriented, 6), dtype=small_dtype)

    q = 0
    for g in range(n_groups):
        flats = _unique_orbit_flats_from_key(group_keys_arr[g], n_patch_types)
        for flat in flats:
            face_ids, cnt, full_face_ids = _boundary_count_sparse_from_flat(flat, n_patch_types)
            lo = int(patch_group_ptr[q])
            hi = int(patch_group_ptr[q + 1])
            if hi - lo != face_ids.size:
                raise RuntimeError(f"Boundary sparse-size mismatch for oriented state {q}: ptr={hi-lo}, actual={face_ids.size}")
            oriented_to_group[q] = g
            orientation_face_keys[q] = full_face_ids
            patch_to_species[lo:hi] = q
            patch_to_small[lo:hi] = face_ids.astype(small_dtype, copy=False)
            m_patch[lo:hi] = cnt.astype(np.float32, copy=False)
            q += 1
        if verbose and ((g + 1) % report_every == 0 or g + 1 == n_groups):
            print(f"[{label}] boundary-oriented pass2 {g + 1}/{n_groups}", flush=True)

    if q != total_oriented:
        raise RuntimeError(f"Internal oriented-state count mismatch: filled {q}, expected {total_oriented}")

    mean_size = float(np.mean(sparse_sizes)) if sparse_sizes else 0.0
    max_size = int(np.max(sparse_sizes)) if sparse_sizes else 0
    min_size = int(np.min(sparse_sizes)) if sparse_sizes else 0
    stats = {
        "boundary_patch_mode": "orientation_resolved_directional_face_patterns",
        "boundary_sparse_entries": int(total_entries),
        "n_oriented_groups": int(total_oriented),
        "orientations_per_group_min": int(np.min(orientation_counts)) if n_groups else 0,
        "orientations_per_group_max": int(np.max(orientation_counts)) if n_groups else 0,
        "orientations_per_group_mean": float(np.mean(orientation_counts)) if n_groups else 0.0,
        "patches_per_group_min": min_size,
        "patches_per_group_max": max_size,
        "patches_per_group_mean": mean_size,
        "m_patch_sum_min": float(min_sum),
        "m_patch_sum_max": float(max_sum),
        "m_patch_sum_expected": 6.0,
        "m_patch_counts_are_integer": True,
    }
    if verbose:
        print(
            f"[{label}] boundary_patch_mode={stats['boundary_patch_mode']} "
            f"canonical_groups={n_groups} oriented_states={total_oriented} "
            f"orient/group[min,max,mean]=({stats['orientations_per_group_min']},"
            f"{stats['orientations_per_group_max']},{stats['orientations_per_group_mean']:.3f}) "
            f"entries={total_entries} per_oriented_state[min,max,mean]=({min_size},{max_size},{mean_size:.3f}) "
            f"m_sum[min,max]=({stats['m_patch_sum_min']:.6g},{stats['m_patch_sum_max']:.6g})",
            flush=True,
        )
    return (
        patch_to_species,
        patch_to_small,
        m_patch,
        patch_group_ptr,
        oriented_to_group,
        orientation_group_ptr,
        orientation_face_keys,
        stats,
    )

def _cube_cache_fingerprint(patches: np.ndarray, *, version: int = 9) -> str:
    h = hashlib.sha1()
    h.update(np.ascontiguousarray(patches, dtype=np.int16).tobytes())
    h.update(str(version).encode("utf-8"))
    return h.hexdigest()[:20]


def _cache_path_is_pickle(path: Path) -> bool:
    name = path.name.lower()
    return name.endswith(".pkl.gz") or name.endswith(".pickle.gz") or name.endswith(".pkl") or name.endswith(".pickle")


def _build_bond_hist_from_bond_pairs(bond_a: np.ndarray, bond_b: np.ndarray, n_patch_types: int) -> np.ndarray:
    bond_a = np.asarray(bond_a)
    bond_b = np.asarray(bond_b)
    if bond_a.shape != bond_b.shape:
        raise ValueError(f"bond_a/b shape mismatch: {bond_a.shape} vs {bond_b.shape}")
    npt = int(n_patch_types)
    npt2 = npt * npt
    total_cfg = int(bond_a.shape[0])
    out = np.zeros((total_cfg, npt2), dtype=np.uint8)
    rows = np.arange(total_cfg, dtype=np.int64)
    for k in range(bond_a.shape[1]):
        idx = bond_a[:, k].astype(np.int64) * npt + bond_b[:, k].astype(np.int64)
        np.add.at(out, (rows, idx), 1)
    return out


def save_cube_cache(cache: dict, path: str | Path) -> None:
    """Save cache either as legacy pickle-gzip or mmap-ready directory.

    If *path* ends with .pkl.gz/.pickle.gz/.pkl/.pickle, write the legacy pickle.
    Otherwise write a directory cache containing .npy arrays plus meta.npz.  The
    spinodal scanner can mmap this directory directly without a large pickle load.
    """
    path = Path(path)
    if _cache_path_is_pickle(path):
        path.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(path, "wb") as fh:
            pickle.dump(cache, fh, protocol=pickle.HIGHEST_PROTOCOL)
        return

    path.mkdir(parents=True, exist_ok=True)
    marker = path / ".ready"
    if marker.exists():
        marker.unlink()

    npt = int(cache["n_patch_types"])
    big_names = [
        "cfg", "species_counts", "group_keys", "group_ptr",
        "patch_to_species", "patch_to_small", "m_patch", "patch_group_ptr",
        "boundary_orbit_mult", "group_orbit_mult",
        "oriented_to_group", "orientation_group_ptr", "orientation_face_keys",
        "cfg_face_ids",
    ]
    for name in big_names:
        if name in cache:
            np.save(path / f"{name}.npy", np.ascontiguousarray(cache[name]))

    if "bond_hist" in cache:
        bond_hist = np.asarray(cache["bond_hist"])
    else:
        bond_hist = _build_bond_hist_from_bond_pairs(cache["bond_a"], cache["bond_b"], npt)
    np.save(path / "bond_hist.npy", np.ascontiguousarray(bond_hist))

    # Keep bond_a/b optional.  The scanner no longer needs them once bond_hist exists,
    # but they are useful for debugging and backward compatibility.
    if "bond_a" in cache:
        np.save(path / "bond_a.npy", np.ascontiguousarray(cache["bond_a"]))
    if "bond_b" in cache:
        np.save(path / "bond_b.npy", np.ascontiguousarray(cache["bond_b"]))

    skip = set(big_names) | {"bond_hist", "bond_a", "bond_b", "cfg_face_ids"}
    meta = {}
    for k, v in cache.items():
        if k in skip:
            continue
        if isinstance(v, np.ndarray):
            meta[k] = np.asarray(v)
        elif isinstance(v, (int, float, bool, np.integer, np.floating, np.bool_)):
            meta[f"_scalar_{k}"] = np.asarray([v])
        elif isinstance(v, str):
            meta[f"_str_{k}"] = np.asarray([v])
    np.savez(path / "meta.npz", **meta)
    marker.write_text("ok\n")


def load_cube_cache(path: str | Path) -> dict:
    path = Path(path)
    if path.is_dir():
        cache = {}
        for npy in sorted(path.glob("*.npy")):
            cache[npy.stem] = np.load(npy, mmap_mode=None)
        meta_path = path / "meta.npz"
        if meta_path.exists():
            meta = np.load(meta_path, allow_pickle=False)
            for k in meta.files:
                if k.startswith("_scalar_"):
                    real_key = k[len("_scalar_"):]
                    val = meta[k][0]
                    if isinstance(val, np.integer):
                        cache[real_key] = int(val)
                    elif isinstance(val, np.floating):
                        cache[real_key] = float(val)
                    elif isinstance(val, np.bool_):
                        cache[real_key] = bool(val)
                    else:
                        cache[real_key] = val
                elif k.startswith("_str_"):
                    cache[k[len("_str_"):]] = str(meta[k][0])
                else:
                    cache[k] = np.asarray(meta[k])
        return cache
    with gzip.open(path, "rb") as fh:
        return pickle.load(fh)


def _build_corner_lookup(patches: np.ndarray):
    n_species = patches.shape[0]
    lookup = []
    triple_values = []
    triple_species = []
    for corner in range(8):
        dirs = CORNER_OUTWARD_DIRS[corner]
        mp = {}
        for s in range(n_species):
            tri = tuple(int(x) for x in patches[s, list(dirs)].tolist())
            mp.setdefault(tri, []).append(s)

        keys = sorted(mp.keys())
        vals = np.asarray(keys, dtype=np.int16)
        sp_lists = [np.asarray(mp[k], dtype=np.int32) for k in keys]
        lookup.append({k: v for k, v in zip(keys, sp_lists)})
        triple_values.append(vals)
        triple_species.append(tuple(sp_lists))
    return tuple(lookup), tuple(triple_values), tuple(triple_species)


def _decode_mixed_radix_chunk(start: int, stop: int, radices: np.ndarray) -> np.ndarray:
    idx = np.arange(start, stop, dtype=np.int64)
    out = np.empty((idx.shape[0], len(radices)), dtype=np.int16)
    tmp = idx.copy()
    for pos in range(len(radices) - 1, -1, -1):
        out[:, pos] = tmp % radices[pos]
        tmp //= radices[pos]
    return out


def _lex_lt_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    batch, ncols = a.shape
    take = np.zeros(batch, dtype=bool)
    undec = np.ones(batch, dtype=bool)
    for j in range(ncols):
        lt = a[:, j] < b[:, j]
        gt = a[:, j] > b[:, j]
        take |= undec & lt
        undec &= ~(lt | gt)
    return take


def _lexicographic_min_over_rotations(cands: np.ndarray):
    batch, nrot, _ = cands.shape
    best = cands[:, 0, :].copy()
    best_idx = np.zeros(batch, dtype=np.int64)
    for k in range(1, nrot):
        cand = cands[:, k, :]
        take = _lex_lt_rows(cand, best)
        if np.any(take):
            best[take] = cand[take]
            best_idx[take] = k
    return best, best_idx


def _build_boundary_chunk_face_flats(triple_values_by_corner, triple_id_chunk: np.ndarray) -> np.ndarray:
    batch = triple_id_chunk.shape[0]
    flat = np.empty((batch, 24), dtype=np.int16)
    for corner in range(8):
        vals = triple_values_by_corner[corner][triple_id_chunk[:, corner]]
        flat[:, CORNER_FACE_SLOT_IDX[corner, 0]] = vals[:, 0]
        flat[:, CORNER_FACE_SLOT_IDX[corner, 1]] = vals[:, 1]
        flat[:, CORNER_FACE_SLOT_IDX[corner, 2]] = vals[:, 2]
    return flat


def _canonicalize_flat_chunk(flat: np.ndarray, n_patch_types: int):
    batch = flat.shape[0]
    nrot = ROT_FLAT_PERMS.shape[0]
    cands = np.empty((batch, nrot, 6), dtype=np.int64)
    for k in range(nrot):
        rot = flat[:, ROT_FLAT_PERMS[k]].reshape(batch, 6, 4).astype(np.int64, copy=False)
        cands[:, k, :] = ((rot[:, :, 0] * n_patch_types + rot[:, :, 1]) * n_patch_types + rot[:, :, 2]) * n_patch_types + rot[:, :, 3]
    return _lexicographic_min_over_rotations(cands)


def _canonical_key_to_flat(key: np.ndarray, n_patch_types: int) -> np.ndarray:
    flat = np.empty(24, dtype=np.int16)
    for face in range(6):
        flat[FACE_FLAT_IDX[face]] = _decode_face_id(int(key[face]), n_patch_types).astype(np.int16)
    return flat


def _orbit_size_from_key(key: np.ndarray, n_patch_types: int) -> int:
    flat = _canonical_key_to_flat(np.asarray(key, dtype=np.int64), n_patch_types)
    seen = set()
    for perm in ROT_FLAT_PERMS:
        rot = flat[perm].reshape(6, 4)
        face_ids = tuple(_encode_face_slots(rot[f], n_patch_types) for f in range(6))
        seen.add(face_ids)
    return len(seen)


def _compatible_cfgs_from_flat(flat: np.ndarray, triple_values_by_corner, triple_species_by_corner) -> np.ndarray:
    flat = np.asarray(flat, dtype=np.int64)
    if flat.shape != (24,):
        raise ValueError(f"flat boundary must have shape (24,), got {flat.shape}")

    species_lists = []
    for corner in range(8):
        tri = tuple(int(x) for x in flat[CORNER_FACE_SLOT_IDX[corner]].tolist())
        vals = triple_values_by_corner[corner]
        hit = np.where(np.all(vals == np.asarray(tri, dtype=vals.dtype), axis=1))[0]
        if hit.size != 1:
            raise RuntimeError(f"No unique compatible species list for corner {corner} and triple {tri}.")
        species_lists.append(triple_species_by_corner[corner][int(hit[0])])

    if any(len(x) == 0 for x in species_lists):
        raise RuntimeError("Encountered empty compatible species list.")
    grids = np.meshgrid(*species_lists, indexing="ij")
    return np.stack(grids, axis=-1).reshape(-1, 8).astype(np.int32, copy=False)


def _unique_orbit_flats_from_key(key: np.ndarray, n_patch_types: int) -> list[np.ndarray]:
    flat0 = _canonical_key_to_flat(np.asarray(key, dtype=np.int64), n_patch_types)
    seen = set()
    out = []
    for perm in ROT_FLAT_PERMS:
        rot = flat0[perm].astype(np.int16, copy=False)
        key_t = tuple(int(x) for x in rot.tolist())
        if key_t not in seen:
            seen.add(key_t)
            out.append(rot.copy())
    return out


def _compatible_cfgs_from_key(key: np.ndarray, triple_values_by_corner, triple_species_by_corner, n_patch_types: int) -> np.ndarray:
    cfg_blocks = []
    for flat in _unique_orbit_flats_from_key(key, n_patch_types):
        cfg_blocks.append(_compatible_cfgs_from_flat(flat, triple_values_by_corner, triple_species_by_corner))

    if not cfg_blocks:
        return np.empty((0, 8), dtype=np.int32)
    if len(cfg_blocks) == 1:
        return cfg_blocks[0].astype(np.int32, copy=False)

    cfg = np.concatenate(cfg_blocks, axis=0).astype(np.int32, copy=False)
    return np.unique(cfg, axis=0).astype(np.int32, copy=False)


def build_cubes_geometry_cache_fast(
    patches: np.ndarray,
    *,
    cache_path: Optional[str] = None,
    cache_dir: str = "cache",
    force_rebuild: bool = False,
    assume_rotation_invariant: bool = True,
    boundary_chunk_size: int = 200_000,
    verbose: bool = True,
) -> dict:
    if not assume_rotation_invariant:
        raise ValueError(
            "This cache assumes rotationally equivalent boundary classes have identical partition sums."
        )

    patches = np.asarray(patches, dtype=np.int64)
    if patches.ndim != 2 or patches.shape[1] != 6:
        raise ValueError(f"patches must have shape (n_species, 6), got {patches.shape}")

    n_species = int(patches.shape[0])
    n_patch_types = int(patches.max()) + 1
    fp = _cube_cache_fingerprint(patches)

    if cache_path is None:
        cache_dir_path = Path(cache_dir)
        cache_dir_path.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir_path / f"cube_geometry_fast_{fp}.pkl.gz"
    cache_path = Path(cache_path)

    if cache_path.exists() and not force_rebuild:
        cache = load_cube_cache(cache_path)
        if cache.get("fingerprint") != fp:
            raise ValueError(
                f"Cache fingerprint mismatch for {cache_path}. Delete it or rebuild."
            )
        return cache

    _, triple_values_by_corner, triple_species_by_corner = _build_corner_lookup(patches)
    radices = np.asarray([len(x) for x in triple_values_by_corner], dtype=np.int64)
    raw_boundary_count = int(np.prod(radices, dtype=np.int64))

    if verbose:
        print(f"[cube-cache] n_species={n_species}, n_patch_types={n_patch_types}")
        print(f"[cube-cache] per-corner triple counts={radices.tolist()}")
        print(f"[cube-cache] raw boundary assignments={raw_boundary_count}")

    seen = {}
    unique_keys = []
    orbit_sizes = []

    for start in range(0, raw_boundary_count, int(boundary_chunk_size)):
        stop = min(raw_boundary_count, start + int(boundary_chunk_size))
        triple_ids = _decode_mixed_radix_chunk(start, stop, radices)
        flat = _build_boundary_chunk_face_flats(triple_values_by_corner, triple_ids)
        keys, _ = _canonicalize_flat_chunk(flat, n_patch_types)
        uniq = np.unique(keys, axis=0)
        for row in uniq:
            key_t = tuple(int(x) for x in row.tolist())
            if key_t not in seen:
                seen[key_t] = len(unique_keys)
                unique_keys.append(np.asarray(row, dtype=np.int64))
                orbit_sizes.append(_orbit_size_from_key(row, n_patch_types))
        if verbose and (stop == raw_boundary_count or stop % (10 * boundary_chunk_size) == 0):
            print(f"[cube-cache] processed {stop}/{raw_boundary_count}; unique_classes={len(unique_keys)}")

    if not unique_keys:
        raise RuntimeError("No cube classes were generated.")

    cfg_blocks = []
    group_ptr = [0]
    boundary_orbit_mult = np.asarray(orbit_sizes, dtype=np.float64)

    for g, key in enumerate(unique_keys):
        cfgs = _compatible_cfgs_from_key(key, triple_values_by_corner, triple_species_by_corner, n_patch_types)
        cfg_blocks.append(cfgs)
        group_ptr.append(group_ptr[-1] + cfgs.shape[0])
        if verbose and (g == 0 or (g + 1) % 500 == 0):
            print(f"[cube-cache] expanded {g + 1}/{len(unique_keys)} classes; reduced_cfgs={group_ptr[-1]}")

    cfg = np.concatenate(cfg_blocks, axis=0).astype(np.int32, copy=False)
    group_ptr = np.asarray(group_ptr, dtype=np.int64)
    n_groups = len(unique_keys)
    group_keys_arr = np.asarray(unique_keys, dtype=np.int64)

    bond_a = np.empty((cfg.shape[0], 12), dtype=np.int16)
    bond_b = np.empty((cfg.shape[0], 12), dtype=np.int16)
    for k, (ci, di, cj, dj) in enumerate(CUBE_BONDS):
        bond_a[:, k] = patches[cfg[:, ci], di]
        bond_b[:, k] = patches[cfg[:, cj], dj]

    species_counts = np.zeros((cfg.shape[0], n_species), dtype=np.uint8)
    rows = np.arange(cfg.shape[0], dtype=np.int64)
    for col in range(8):
        species_counts[rows, cfg[:, col]] += 1

    # Full undirected face-pattern basis. Do not shrink this to the face types
    # appearing in canonical representatives; otherwise n_ft depends on particle
    # geometry and rotational canonicalization leaks into the association basis.
    face_radix, small_face_ids, small_face_dirs, small_face_slots = _build_directional_face_basis(n_patch_types)

    patch_to_species, patch_to_small, m_patch, patch_group_ptr, oriented_to_group, orientation_group_ptr, orientation_face_keys, patch_stats = _build_orientation_resolved_patch_arrays(
        group_keys_arr,
        n_patch_types,
        group_index_dtype=np.int32,
        verbose=verbose,
        label="cube-cache",
    )

    cache = {
        "version": 9,
        "fingerprint": fp,
        "patches": patches.copy(),
        "n_species": n_species,
        "n_patch_types": n_patch_types,
        "cfg": cfg,
        "group_ptr": group_ptr,
        "group_orbit_mult": np.ones(n_groups, dtype=np.float64),
        "boundary_orbit_mult": boundary_orbit_mult,
        "cfg_includes_full_orbit": True,
        "bond_a": bond_a,
        "bond_b": bond_b,
        "species_counts": species_counts,
        "group_keys": group_keys_arr,
        "n_groups": n_groups,
        "patch_to_species": patch_to_species,
        "patch_to_small": patch_to_small,
        "m_patch": m_patch,
        "patch_group_ptr": patch_group_ptr,
        "oriented_to_group": oriented_to_group,
        "orientation_group_ptr": orientation_group_ptr,
        "orientation_face_keys": orientation_face_keys,
        **patch_stats,
        "has_cfg_face_ids": False,
        "face_basis_mode": "directional_6xM4_deterministic_opposite_faces",
        "expected_n_ft": int(6 * face_radix),
        "small_face_ids": small_face_ids.astype(np.int64),
        "small_face_dirs": small_face_dirs.astype(np.int64),
        "small_face_slots": small_face_slots,
        "raw_boundary_count": raw_boundary_count,
        "unique_classes": n_groups,
        "reduced_cfg_count": int(cfg.shape[0]),
        "cache_path": str(cache_path),
    }

    save_cube_cache(cache, cache_path)
    if verbose:
        print(f"[cube-cache] saved -> {cache_path}")
        print(f"[cube-cache] unique_classes={n_groups}, reduced_cfgs={cfg.shape[0]}")
        print(f"[cube-cache] face_basis_mode=directional_6xM4_deterministic_opposite_faces n_ft={len(small_face_ids)} expected={6 * face_radix}")
    return cache


def cubes_from_cache(cache: dict, J: np.ndarray, mu: Optional[np.ndarray] = None):
    J = np.asarray(J, dtype=np.float64)
    n_species = int(cache["n_species"])
    n_patch_types = int(cache["n_patch_types"])
    if J.shape != (n_patch_types, n_patch_types):
        raise ValueError(f"J must have shape ({n_patch_types}, {n_patch_types}), got {J.shape}")

    mu_vec = np.zeros(n_species, dtype=np.float64) if mu is None else np.asarray(mu, dtype=np.float64)
    if mu_vec.shape != (n_species,):
        raise ValueError(f"mu must have shape ({n_species},), got {mu_vec.shape}")

    cfg = np.asarray(cache["cfg"], dtype=np.int64)
    group_ptr = np.asarray(cache["group_ptr"], dtype=np.int64)
    group_orbit_mult = np.asarray(cache["group_orbit_mult"], dtype=np.float64)
    bond_a = np.asarray(cache["bond_a"], dtype=np.int64)
    bond_b = np.asarray(cache["bond_b"], dtype=np.int64)
    species_counts = np.asarray(cache["species_counts"], dtype=np.float64)
    n_groups = int(cache["n_groups"])

    E = J[bond_a, bond_b].sum(axis=1)
    mu_sum = mu_vec[cfg].sum(axis=1)
    Eeff = E - mu_sum

    cube_to_species = np.empty((n_groups, n_species), dtype=np.float64)
    intra_bonds = np.empty(n_groups, dtype=np.float64)
    cube_configs = np.empty((n_groups, 8), dtype=np.int32)

    for g in range(n_groups):
        lo = int(group_ptr[g])
        hi = int(group_ptr[g + 1])
        E_slice = Eeff[lo:hi]
        Emin = float(E_slice.min())
        wcfg = np.exp(-(E_slice - Emin))
        z = max(float(wcfg.sum()), 1e-300)
        cube_to_species[g] = (wcfg[:, None] * species_counts[lo:hi]).sum(axis=0) / z
        intra_bonds[g] = float((wcfg * E_slice).sum() / z)
        cube_configs[g] = cfg[lo + int(np.argmin(E_slice))]

    slots = np.asarray(cache["small_face_slots"], dtype=np.int64)
    dirs = np.asarray(cache.get("small_face_dirs"), dtype=np.int64)
    if dirs.shape[0] != slots.shape[0]:
        raise RuntimeError("Direction-resolved cache is missing valid small_face_dirs.")
    eps_small = np.zeros((slots.shape[0], slots.shape[0]), dtype=np.float64)
    for d in range(6):
        ii = np.where(dirs == d)[0]
        jj = np.where(dirs == int(OPPOSITE_FACE[d]))[0]
        if ii.size == 0 or jj.size == 0:
            continue
        block = np.zeros((ii.size, jj.size), dtype=np.float64)
        si = slots[ii]
        sj = slots[jj]
        for k in range(4):
            block += J[si[:, k][:, None], sj[:, k][None, :]]
        eps_small[np.ix_(ii, jj)] = block

    if "oriented_to_group" in cache:
        # Association states are explicit orientations alpha=(g,R).  Use the
        # per-orientation internal free energy convention: Z_alpha = Z_g / |O_g|.
        oriented_to_group = np.asarray(cache["oriented_to_group"], dtype=np.int64)
        boundary_mult = np.asarray(cache.get("boundary_orbit_mult", np.ones(n_groups)), dtype=np.float64)
        # This helper historically returns intra_bonds=<E>; keep that value for
        # compatibility but expand it to oriented states.  New scan code should use
        # -logZ from cubes_from_cache_fast instead.
        intra_bonds_out = intra_bonds[oriented_to_group]
        cube_to_species_out = cube_to_species[oriented_to_group]
        cube_configs_out = cube_configs[oriented_to_group]
        mult_arr = np.ones(oriented_to_group.shape[0], dtype=np.float64)
    else:
        intra_bonds_out = intra_bonds
        cube_to_species_out = cube_to_species
        cube_configs_out = cube_configs
        mult_arr = group_orbit_mult.copy()

    return (
        eps_small,
        intra_bonds_out,
        cube_to_species_out,
        np.asarray(cache["m_patch"], dtype=np.float64).copy(),
        np.asarray(cache["patch_to_species"], dtype=np.int64).copy(),
        np.asarray(cache["patch_to_small"], dtype=np.int64).copy(),
        cube_configs_out,
        mult_arr,
    )


def _min_uint_dtype(max_value: int):
    max_value = int(max_value)
    if max_value <= np.iinfo(np.uint8).max:
        return np.uint8
    if max_value <= np.iinfo(np.uint16).max:
        return np.uint16
    if max_value <= np.iinfo(np.uint32).max:
        return np.uint32
    return np.uint64


def _save_npz(path: str | Path, **arrays) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def _load_npz(path: str | Path):
    return np.load(path, allow_pickle=False)


def _choose_prefix_len(radices: np.ndarray, target_shards: int) -> int:
    prod = 1
    for i, r in enumerate(radices.tolist(), start=1):
        prod *= int(r)
        if prod >= int(target_shards):
            return i
    return len(radices)


def _decode_prefix_index(prefix_index: int, prefix_radices: np.ndarray) -> np.ndarray:
    out = np.empty(len(prefix_radices), dtype=np.int16)
    x = int(prefix_index)
    for i in range(len(prefix_radices) - 1, -1, -1):
        r = int(prefix_radices[i])
        out[i] = x % r
        x //= r
    return out


def _iter_assigned_prefixes(n_prefix: int, shard_id: int, n_shards: int) -> Iterable[int]:
    for p in range(int(shard_id), int(n_prefix), int(n_shards)):
        yield p


def key_shard_worker(
    patches_path: str,
    out_dir: str,
    shard_id: int,
    n_shards: int,
    *,
    target_shards: int = 100,
    boundary_chunk_size: int = 200_000,
    prefix_len: Optional[int] = None,
    verbose: bool = True,
) -> str:
    patches = load_patches_npy(patches_path)
    _, triple_values_by_corner, _ = _build_corner_lookup(patches)
    radices = np.asarray([len(x) for x in triple_values_by_corner], dtype=np.int64)
    n_patch_types = int(patches.max()) + 1

    if prefix_len is None:
        prefix_len = _choose_prefix_len(radices, target_shards)
    prefix_radices = radices[:prefix_len]
    suffix_radices = radices[prefix_len:]
    n_prefix = int(np.prod(prefix_radices, dtype=np.int64)) if prefix_len > 0 else 1
    suffix_count = int(np.prod(suffix_radices, dtype=np.int64)) if len(suffix_radices) > 0 else 1

    local_keys = set()
    processed = 0
    for prefix_index in _iter_assigned_prefixes(n_prefix, shard_id, n_shards):
        prefix = _decode_prefix_index(prefix_index, prefix_radices) if prefix_len > 0 else np.empty(0, dtype=np.int16)
        for start in range(0, suffix_count, int(boundary_chunk_size)):
            stop = min(suffix_count, start + int(boundary_chunk_size))
            if len(suffix_radices) > 0:
                suffix = _decode_mixed_radix_chunk(start, stop, suffix_radices)
                triple_ids = np.empty((suffix.shape[0], len(radices)), dtype=np.int16)
                if prefix_len > 0:
                    triple_ids[:, :prefix_len] = prefix[None, :]
                triple_ids[:, prefix_len:] = suffix
            else:
                triple_ids = prefix.reshape(1, -1).astype(np.int16, copy=False)
            flat = _build_boundary_chunk_face_flats(triple_values_by_corner, triple_ids)
            keys, _ = _canonicalize_flat_chunk(flat, n_patch_types)
            uniq = np.unique(keys, axis=0)
            for row in uniq:
                local_keys.add(tuple(int(x) for x in row.tolist()))
            processed += (stop - start)

    arr = np.asarray(sorted(local_keys), dtype=np.int64)
    out_path = Path(out_dir) / f"keys_shard_{int(shard_id):04d}.npz"
    _save_npz(
        out_path,
        keys=arr,
        processed=np.asarray([processed], dtype=np.int64),
        radices=radices,
        shard_id=np.asarray([int(shard_id)], dtype=np.int64),
        n_shards=np.asarray([int(n_shards)], dtype=np.int64),
    )
    if verbose:
        print(f"[cube-keys] shard={shard_id}/{n_shards} processed={processed} unique_local={len(arr)} -> {out_path}")
    return str(out_path)


def _exact_shard_files(directory: str | Path, prefix: str, n_shards: Optional[int]) -> list[Path]:
    directory = Path(directory)
    if n_shards is None:
        files = sorted(directory.glob(f"{prefix}_*.npz"))
        if not files:
            raise FileNotFoundError(f"No {prefix}_*.npz shards in {directory}")
        return files

    files = [directory / f"{prefix}_{i:04d}.npz" for i in range(int(n_shards))]
    missing = [str(f) for f in files if not f.exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing {len(missing)} expected {prefix} shards for n_shards={n_shards}. "
            f"First missing files: {missing[:10]}"
        )
    return files


def merge_key_shards(
    patches_path: str,
    keys_dir: str,
    merged_keys_path: str,
    *,
    n_shards: Optional[int] = None,
    verbose: bool = True,
) -> str:
    patches = load_patches_npy(patches_path)
    n_patch_types = int(patches.max()) + 1
    files = _exact_shard_files(keys_dir, "keys_shard", n_shards)

    all_keys = []
    total_processed = 0
    for fp in files:
        z = _load_npz(fp)
        if "n_shards" in z.files and n_shards is not None and int(z["n_shards"][0]) != int(n_shards):
            raise RuntimeError(
                f"Shard {fp} was generated with n_shards={int(z['n_shards'][0])}, "
                f"but this merge expects n_shards={n_shards}. Delete stale shards or rerun with CLEAN_BUILD=1."
            )
        all_keys.append(np.asarray(z["keys"], dtype=np.int64))
        total_processed += int(z["processed"][0])

    merged = np.unique(np.concatenate(all_keys, axis=0), axis=0)
    orbit_sizes = np.asarray([_orbit_size_from_key(row, n_patch_types) for row in merged], dtype=np.float64)
    _save_npz(
        merged_keys_path,
        group_keys=merged,
        orbit_sizes=orbit_sizes,
        total_processed=np.asarray([total_processed], dtype=np.int64),
        n_key_shards=np.asarray([len(files)], dtype=np.int64),
    )
    if verbose:
        print(f"[cube-keys-merge] shards={len(files)} unique_classes={len(merged)} total_processed={total_processed} -> {merged_keys_path}")
    return str(merged_keys_path)


def expand_group_shard(
    patches_path: str,
    merged_keys_path: str,
    out_dir: str,
    shard_id: int,
    n_shards: int,
    *,
    verbose: bool = True,
) -> str:
    patches = load_patches_npy(patches_path)
    _, triple_values_by_corner, triple_species_by_corner = _build_corner_lookup(patches)
    n_patch_types = int(patches.max()) + 1

    z = _load_npz(merged_keys_path)
    group_keys = np.asarray(z["group_keys"], dtype=np.int64)
    orbit_sizes = np.asarray(z["orbit_sizes"], dtype=np.float64)
    n_groups = len(group_keys)

    group_ids = np.arange(shard_id, n_groups, n_shards, dtype=np.int64)
    cfg_blocks: List[np.ndarray] = []
    group_ptr = [0]
    local_keys = []
    local_orbit = []

    for g in group_ids.tolist():
        key = group_keys[g]
        cfgs = _compatible_cfgs_from_key(key, triple_values_by_corner, triple_species_by_corner, n_patch_types)
        cfg_blocks.append(cfgs.astype(np.int32, copy=False))
        group_ptr.append(group_ptr[-1] + cfgs.shape[0])
        local_keys.append(key)
        local_orbit.append(orbit_sizes[g])

    cfg = np.concatenate(cfg_blocks, axis=0) if cfg_blocks else np.empty((0, 8), dtype=np.int32)
    out_path = Path(out_dir) / f"expand_shard_{int(shard_id):04d}.npz"
    _save_npz(
        out_path,
        group_ids=group_ids,
        group_keys=np.asarray(local_keys, dtype=np.int64),
        orbit_sizes=np.asarray(local_orbit, dtype=np.float64),
        group_ptr=np.asarray(group_ptr, dtype=np.int64),
        cfg=cfg,
        shard_id=np.asarray([int(shard_id)], dtype=np.int64),
        n_shards=np.asarray([int(n_shards)], dtype=np.int64),
    )
    if verbose:
        print(f"[cube-expand] shard={shard_id}/{n_shards} groups={len(group_ids)} reduced_cfgs={len(cfg)} -> {out_path}")
    return str(out_path)


def merge_expand_shards(
    patches_path: str,
    merged_keys_path: str,
    expand_dir: str,
    final_cache_path: str,
    *,
    n_shards: Optional[int] = None,
    store_cfg_face_ids: bool = False,
    post_chunk_rows: int = 500_000,
    verbose: bool = True,
) -> str:
    patches = load_patches_npy(patches_path)
    n_species = int(patches.shape[0])
    n_patch_types = int(patches.max()) + 1
    fp = _cube_cache_fingerprint(patches)

    zkeys = _load_npz(merged_keys_path)
    group_keys_all = np.asarray(zkeys["group_keys"], dtype=np.int64)
    orbit_sizes_all = np.asarray(zkeys["orbit_sizes"], dtype=np.float64)
    n_groups = len(group_keys_all)

    files = _exact_shard_files(expand_dir, "expand_shard", n_shards)

    # Pass 1: collect group sizes only. This avoids holding all cfg blocks in RAM.
    group_sizes = np.full(n_groups, -1, dtype=np.int64)
    group_source = np.full(n_groups, -1, dtype=np.int64)
    file_group_ids = []
    for file_idx, fpz in enumerate(files):
        z = _load_npz(fpz)
        if "n_shards" in z.files and n_shards is not None and int(z["n_shards"][0]) != int(n_shards):
            raise RuntimeError(
                f"Shard {fpz} was generated with n_shards={int(z['n_shards'][0])}, "
                f"but this merge expects n_shards={n_shards}. Delete stale shards or rerun with CLEAN_BUILD=1."
            )
        gids = np.asarray(z["group_ids"], dtype=np.int64)
        gptr = np.asarray(z["group_ptr"], dtype=np.int64)
        if gids.size + 1 != gptr.size:
            raise RuntimeError(f"Malformed shard {fpz}: len(group_ids)={gids.size}, len(group_ptr)={gptr.size}")
        if gids.size and (gids.min() < 0 or gids.max() >= n_groups):
            raise RuntimeError(
                f"Shard {fpz} contains group ids outside merged key range [0,{n_groups}). "
                f"Delete stale shards or rerun with CLEAN_BUILD=1."
            )
        sizes = np.diff(gptr).astype(np.int64, copy=False)
        for g, sz in zip(gids.tolist(), sizes.tolist()):
            if group_sizes[g] != -1:
                prev = files[int(group_source[g])]
                raise RuntimeError(
                    f"Duplicate expanded group {g} encountered in {fpz}; already present in {prev}. "
                    f"This almost always means EXPAND_DIR contains stale shards from a run with a different --n-shards. "
                    f"Delete {Path(expand_dir)} or rerun the build script with CLEAN_BUILD=1."
                )
            group_sizes[g] = int(sz)
            group_source[g] = file_idx
        file_group_ids.append(gids)

    missing = np.where(group_sizes < 0)[0]
    if missing.size:
        raise RuntimeError(f"Missing expanded groups, examples: {missing[:20].tolist()}")

    group_ptr = np.empty(n_groups + 1, dtype=np.int64)
    group_ptr[0] = 0
    np.cumsum(group_sizes, out=group_ptr[1:])
    total_cfg = int(group_ptr[-1])

    cfg_dtype = _min_uint_dtype(max(n_species - 1, 0))
    patch_dtype = _min_uint_dtype(max(n_patch_types - 1, 0))
    group_index_dtype = _min_uint_dtype(max(n_groups - 1, 0))

    # Pass 2: allocate once and stream shard cfg blocks directly into final positions.
    cfg = np.empty((total_cfg, 8), dtype=cfg_dtype)
    for fpz, gids in zip(files, file_group_ids):
        z = _load_npz(fpz)
        gptr = np.asarray(z["group_ptr"], dtype=np.int64)
        shard_cfg = np.asarray(z["cfg"], dtype=np.int64)
        for i, g in enumerate(gids.tolist()):
            src_lo = int(gptr[i])
            src_hi = int(gptr[i + 1])
            dst_lo = int(group_ptr[g])
            dst_hi = int(group_ptr[g + 1])
            if dst_hi - dst_lo != src_hi - src_lo:
                raise RuntimeError(f"Size mismatch for group {g}: dst={dst_hi - dst_lo}, src={src_hi - src_lo}")
            cfg[dst_lo:dst_hi] = shard_cfg[src_lo:src_hi].astype(cfg_dtype, copy=False)

    # Full undirected face-pattern basis. This must be geometry-independent:
    # n_ft = n_patch_types**4 for all particle shapes with the same patch alphabet.
    face_radix, small_face_ids, small_face_dirs, small_face_slots = _build_directional_face_basis(n_patch_types)

    # Pass 3: derive bond hist inputs and species counts in chunks.
    # Do NOT materialize cfg.astype(int64) for the whole array. For 40M+ configs that
    # temporary is several GB and is the usual cause of OOM during merge-cache.
    bond_a = np.empty((total_cfg, 12), dtype=patch_dtype)
    bond_b = np.empty((total_cfg, 12), dtype=patch_dtype)
    species_counts = np.zeros((total_cfg, n_species), dtype=np.uint8)

    small_dtype = _min_uint_dtype(6 * face_radix - 1)
    cfg_face_ids = None
    if store_cfg_face_ids:
        cfg_face_ids = np.empty((total_cfg, 6), dtype=small_dtype)

    post_chunk_rows = max(1, int(post_chunk_rows))
    if verbose:
        print(f"[cube-cache-final] postprocess total_cfg={total_cfg} chunk_rows={post_chunk_rows} store_cfg_face_ids={store_cfg_face_ids}", flush=True)

    for lo in range(0, total_cfg, post_chunk_rows):
        hi = min(total_cfg, lo + post_chunk_rows)
        cfg_chunk = cfg[lo:hi].astype(np.int64, copy=False)

        for k, (ci, di, cj, dj) in enumerate(CUBE_BONDS):
            bond_a[lo:hi, k] = patches[cfg_chunk[:, ci], di].astype(patch_dtype, copy=False)
            bond_b[lo:hi, k] = patches[cfg_chunk[:, cj], dj].astype(patch_dtype, copy=False)

        sc = species_counts[lo:hi]
        rows_local = np.arange(hi - lo, dtype=np.int64)
        for col in range(8):
            sc[rows_local, cfg_chunk[:, col]] += 1

        if cfg_face_ids is not None:
            for face_dir in range(6):
                corners = FACE_SLOT_CORNERS[face_dir]
                s0 = patches[cfg_chunk[:, int(corners[0])], face_dir].astype(np.int64, copy=False)
                s1 = patches[cfg_chunk[:, int(corners[1])], face_dir].astype(np.int64, copy=False)
                s2 = patches[cfg_chunk[:, int(corners[2])], face_dir].astype(np.int64, copy=False)
                s3 = patches[cfg_chunk[:, int(corners[3])], face_dir].astype(np.int64, copy=False)
                face_id = ((s0 * n_patch_types + s1) * n_patch_types + s2) * n_patch_types + s3
                cfg_face_ids[lo:hi, face_dir] = (face_dir * face_radix + face_id).astype(small_dtype, copy=False)

        if verbose and (hi == total_cfg or hi % (10 * post_chunk_rows) == 0):
            print(f"[cube-cache-final] postprocessed {hi}/{total_cfg}", flush=True)

    patch_to_species, patch_to_small, m_patch, patch_group_ptr, oriented_to_group, orientation_group_ptr, orientation_face_keys, patch_stats = _build_orientation_resolved_patch_arrays(
        group_keys_all,
        n_patch_types,
        group_index_dtype=group_index_dtype,
        verbose=verbose,
        label="cube-cache-final",
    )

    cache = {
        "version": 9,
        "fingerprint": fp,
        "patches": patches.copy(),
        "n_species": n_species,
        "n_patch_types": n_patch_types,
        "cfg": cfg,
        "group_ptr": group_ptr,
        "group_orbit_mult": np.ones(n_groups, dtype=np.float64),
        "boundary_orbit_mult": orbit_sizes_all,
        "cfg_includes_full_orbit": True,
        "bond_a": bond_a,
        "bond_b": bond_b,
        "species_counts": species_counts,
        "group_keys": group_keys_all,
        "n_groups": n_groups,
        "patch_to_species": patch_to_species,
        "patch_to_small": patch_to_small,
        "m_patch": m_patch,
        "patch_group_ptr": patch_group_ptr,
        "oriented_to_group": oriented_to_group,
        "orientation_group_ptr": orientation_group_ptr,
        "orientation_face_keys": orientation_face_keys,
        **patch_stats,
        "has_cfg_face_ids": bool(cfg_face_ids is not None),
        "face_basis_mode": "directional_6xM4_deterministic_opposite_faces",
        "expected_n_ft": int(6 * face_radix),
        "small_face_ids": small_face_ids.astype(np.int64),
        "small_face_dirs": small_face_dirs.astype(np.int64),
        "small_face_slots": small_face_slots,
        "raw_boundary_count": -1,
        "unique_classes": n_groups,
        "reduced_cfg_count": total_cfg,
        "cache_path": str(final_cache_path),
    }
    if cfg_face_ids is not None:
        cache["cfg_face_ids"] = cfg_face_ids

    save_cube_cache(cache, final_cache_path)
    if verbose:
        print(f"[cube-cache-final] unique_classes={n_groups}, reduced_cfgs={total_cfg} -> {final_cache_path}")
        print(f"[cube-cache-final] face_basis_mode=directional_6xM4_deterministic_opposite_faces n_ft={len(small_face_ids)} expected={6 * face_radix}")
        cft_dtype = None if cfg_face_ids is None else cfg_face_ids.dtype
        print(f"[cube-cache-final] dtypes cfg={cfg.dtype} bond={bond_a.dtype} patch_to_species={patch_to_species.dtype} patch_to_small={patch_to_small.dtype} cfg_face_ids={cft_dtype}")
    return str(final_cache_path)


def cmd_build_patches(args) -> None:
    base_patch = parse_base_patch(args.base_patch)
    patches = build_patches_with_vacancy(base_patch, vacancy_type=args.vacancy_type)
    save_patches_npy(args.out, patches)
    print("patches.shape =", patches.shape)
    print(patches)
    print("saved =", str(args.out))


def cmd_ensure_cache(args) -> None:
    patches = load_patches_npy(args.patches)
    cache = build_cubes_geometry_cache_fast(
        patches,
        cache_path=args.cache_path,
        force_rebuild=args.force_rebuild,
        boundary_chunk_size=args.boundary_chunk_size,
        verbose=True,
    )
    print("cache_path =", str(args.cache_path))
    print("n_groups =", cache["n_groups"])
    print("reduced_cfg_count =", cache["reduced_cfg_count"])


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Cube cache builder and cache utilities.")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("build-patches", help="Generate rotated species plus explicit vacancy.")
    s.add_argument("--base-patch", required=True, help="Comma-separated 6-entry base patch, e.g. 1,1,1,0,0,0")
    s.add_argument("--vacancy-type", type=int, default=None, help="Patch type used for the explicit vacancy species.")
    s.add_argument("--out", required=True)

    s = sub.add_parser("ensure-cache", help="Sequential one-shot cache build.")
    s.add_argument("--patches", required=True)
    s.add_argument("--cache-path", required=True)
    s.add_argument("--boundary-chunk-size", type=int, default=200_000)
    s.add_argument("--force-rebuild", action="store_true")

    s = sub.add_parser("keys-shard")
    s.add_argument("--patches", required=True)
    s.add_argument("--out-dir", required=True)
    s.add_argument("--shard-id", type=int, required=True)
    s.add_argument("--n-shards", type=int, required=True)
    s.add_argument("--target-shards", type=int, default=100)
    s.add_argument("--prefix-len", type=int, default=None)
    s.add_argument("--boundary-chunk-size", type=int, default=200_000)

    s = sub.add_parser("merge-keys")
    s.add_argument("--patches", required=True)
    s.add_argument("--keys-dir", required=True)
    s.add_argument("--merged-keys", required=True)
    s.add_argument("--n-shards", type=int, default=None)

    s = sub.add_parser("expand-shard")
    s.add_argument("--patches", required=True)
    s.add_argument("--merged-keys", required=True)
    s.add_argument("--out-dir", required=True)
    s.add_argument("--shard-id", type=int, required=True)
    s.add_argument("--n-shards", type=int, required=True)

    s = sub.add_parser("merge-cache")
    s.add_argument("--patches", required=True)
    s.add_argument("--merged-keys", required=True)
    s.add_argument("--expand-dir", required=True)
    s.add_argument("--final-cache", required=True)
    s.add_argument("--n-shards", type=int, default=None)
    s.add_argument("--store-cfg-face-ids", action="store_true", help="Store cfg_face_ids(total_cfg,6). Useful for debugging face-pattern ids, but costs memory/disk.")
    s.add_argument("--post-chunk-rows", type=int, default=500_000, help="Chunk size for merge-cache postprocessing to avoid int64 cfg temporaries.")

    return p


def main() -> None:
    args = build_parser().parse_args()
    if args.cmd == "build-patches":
        cmd_build_patches(args)
    elif args.cmd == "ensure-cache":
        cmd_ensure_cache(args)
    elif args.cmd == "keys-shard":
        key_shard_worker(
            patches_path=args.patches,
            out_dir=args.out_dir,
            shard_id=args.shard_id,
            n_shards=args.n_shards,
            target_shards=args.target_shards,
            prefix_len=args.prefix_len,
            boundary_chunk_size=args.boundary_chunk_size,
            verbose=True,
        )
    elif args.cmd == "merge-keys":
        merge_key_shards(
            patches_path=args.patches,
            keys_dir=args.keys_dir,
            merged_keys_path=args.merged_keys,
            n_shards=args.n_shards,
            verbose=True,
        )
    elif args.cmd == "expand-shard":
        expand_group_shard(
            patches_path=args.patches,
            merged_keys_path=args.merged_keys,
            out_dir=args.out_dir,
            shard_id=args.shard_id,
            n_shards=args.n_shards,
            verbose=True,
        )
    elif args.cmd == "merge-cache":
        merge_expand_shards(
            patches_path=args.patches,
            merged_keys_path=args.merged_keys,
            expand_dir=args.expand_dir,
            final_cache_path=args.final_cache,
            n_shards=args.n_shards,
            store_cfg_face_ids=args.store_cfg_face_ids,
            post_chunk_rows=args.post_chunk_rows,
            verbose=True,
        )
    else:
        raise SystemExit(f"Unknown command: {args.cmd}")


if __name__ == "__main__":
    main()
