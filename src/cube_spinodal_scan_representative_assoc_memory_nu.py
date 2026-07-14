#!/usr/bin/env python
"""Parallel sparse cube spinodal / binodal scan – representative-boundary registry-mode version with strict feasible-phi filtering.

Memory design (target: <3 GB per process with 80 processes on 250 GB)
======================================================================
1. Bond histogram precomputation: at prepare-mmap time, we precompute
   bond_hist[i, a*npt+b] = count of bonds with patch types (a,b) for
   config i.  This is (total_cfg, npt²) uint8 ≈ 108 MB for 12M configs.
   Stored as .npy and mmap'd read-only.

2. Energy = bond_hist @ J.ravel() — a single matrix-vector product on
   a mmap'd uint8 array.  NO fancy indexing, NO int64 expansion, NO
   fragmentation.  Temporary = (chunk_rows,) float64.

3. species_counts stays mmap'd uint8.  We read slices directly into a
   float64 buffer that is PRE-ALLOCATED and REUSED across calls.

4. gc.collect() + malloc_trim() after each state point to release any
   residual arena pages.

5. All cubes_from_cache output arrays are pre-allocated once and reused.
"""
from __future__ import annotations

import argparse
import ctypes
import gc
import gzip
import json
import os
import pickle
import time
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from scipy.spatial import ConvexHull

try:
    from numba import njit as _njit
except ImportError:
    def _njit(*args, **kwargs):
        def _wrap(fn):
            return fn
        if args and callable(args[0]):
            return args[0]
        return _wrap

# ---------------------------------------------------------------------------
# Import from representative-orientation cache builder
# ---------------------------------------------------------------------------
try:
    from cube_cache_builder_streaming_directional import (
        load_patches_npy,
        OPPOSITE_FACE,
        CUBE_BONDS,
    )
except ImportError:
    try:
        from cube_cache_builder_streaming_representative import (
            load_patches_npy,
            OPPOSITE_FACE,
            CUBE_BONDS,
        )
    except ImportError:
        # Fallbacks keep this scanner usable with older cache-builder filenames.
        try:
            from cube_cache_builder_streaming_registryavg import (
                load_patches_npy,
                OPPOSITE_FACE,
                CUBE_BONDS,
            )
        except ImportError:
            from cube_cache_builder_fixed import (
                load_patches_npy,
                OPPOSITE_FACE,
                CUBE_BONDS,
            )


# ===================================================================
# Memory management helpers
# ===================================================================

def _release_memory():
    """Force Python GC and ask glibc to return free pages to OS."""
    gc.collect()
    try:
        _libc = ctypes.CDLL("libc.so.6")
        _libc.malloc_trim(0)
    except (OSError, AttributeError):
        pass


def _pin_malloc_mmap_threshold():
    """Force glibc to use mmap for allocations > 256 KB.

    The Broyden/Newton fallback solver calls _eval_residual many times, each
    allocating ~2 MB arrays via Numba's NRT (which calls malloc).
    glibc's dynamic mmap threshold drifts upward, pushing these onto
    the heap where they cause permanent RSS growth.  Pinning it low
    ensures every >256 KB allocation uses mmap and is returned to the
    OS immediately on free.
    """
    try:
        _libc = ctypes.CDLL("libc.so.6")
        M_MMAP_THRESHOLD = -3  # glibc mallopt param
        M_TRIM_THRESHOLD = -1
        M_MMAP_MAX = -4
        _libc.mallopt(M_MMAP_THRESHOLD, 256 * 1024)   # 256 KB
        _libc.mallopt(M_TRIM_THRESHOLD, 128 * 1024)    # trim aggressively
        _libc.mallopt(M_MMAP_MAX, 65536)               # allow many mmap regions
    except (OSError, AttributeError):
        pass


# ===================================================================
# Memory-mapped cache: prepare + load
# ===================================================================

def _link_or_copy(src: Path, dst: Path) -> None:
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        os.symlink(src, dst)
    except OSError:
        import shutil
        shutil.copy2(src, dst)


def prepare_mmap_cache(cache_pkl_gz: str, mmap_dir: str, *, verbose: bool = True) -> str:
    """Prepare mmap-friendly arrays for workers.

    Supports two cache formats:
      1. old monolithic .pkl.gz cache: load it and build bond_hist.npy;
      2. new directory cache: link/copy existing .npy arrays and meta.npz.

    The directory cache avoids the slow gzip/pickle final-cache step and already
    stores bond_hist.npy, so this function becomes nearly instantaneous.
    """
    mmap_dir = Path(mmap_dir)
    mmap_dir.mkdir(parents=True, exist_ok=True)
    marker = mmap_dir / ".ready"

    if marker.exists():
        if verbose:
            print(f"[prepare-mmap] already prepared: {mmap_dir}")
        return str(mmap_dir)

    src = Path(cache_pkl_gz)
    if src.is_dir():
        if verbose:
            print(f"[prepare-mmap] using directory cache: {src}")
        required = ["cfg.npy", "bond_hist.npy", "species_counts.npy", "group_keys.npy", "group_ptr.npy", "meta.npz"]
        missing = [x for x in required if not (src / x).exists()]
        if missing:
            raise FileNotFoundError(f"Directory cache {src} is missing required files: {missing}")
        for fp in sorted(src.glob("*.npy")):
            _link_or_copy(fp.resolve(), mmap_dir / fp.name)
            if verbose:
                print(f"[prepare-mmap]   linked {fp.name}")
        _link_or_copy((src / "meta.npz").resolve(), mmap_dir / "meta.npz")
        marker.write_text("ok\n")
        if verbose:
            print(f"[prepare-mmap] directory cache ready: {mmap_dir}")
        return str(mmap_dir)

    if verbose:
        print(f"[prepare-mmap] loading legacy pickle cache {cache_pkl_gz} ...")
    with gzip.open(cache_pkl_gz, "rb") as fh:
        cache = pickle.load(fh)

    n_patch_types = int(cache["n_patch_types"])
    bond_a = np.asarray(cache["bond_a"])
    bond_b = np.asarray(cache["bond_b"])
    total_cfg = bond_a.shape[0]

    npt2 = n_patch_types * n_patch_types
    if verbose:
        print(f"[prepare-mmap] computing bond_hist ({total_cfg}, {npt2}) ...")
    bond_hist = np.zeros((total_cfg, npt2), dtype=np.uint8)
    rows = np.arange(total_cfg, dtype=np.int64)
    for k in range(12):
        idx = bond_a[:, k].astype(np.int32) * n_patch_types + bond_b[:, k].astype(np.int32)
        np.add.at(bond_hist, (rows, idx), 1)

    big_arrays = {
        "cfg": np.ascontiguousarray(cache["cfg"]),
        "bond_hist": bond_hist,
        "species_counts": np.ascontiguousarray(cache["species_counts"]),
        "group_keys": np.ascontiguousarray(cache["group_keys"]),
        "group_ptr": np.ascontiguousarray(cache["group_ptr"], dtype=np.int64),
    }
    for name in ("patch_to_species", "patch_to_small", "m_patch", "patch_group_ptr", "boundary_orbit_mult", "group_orbit_mult"):
        if name in cache:
            big_arrays[name] = np.ascontiguousarray(cache[name])
    for name, arr in big_arrays.items():
        path = mmap_dir / f"{name}.npy"
        np.save(path, arr)
        if verbose:
            print(f"[prepare-mmap]   {name}: shape={arr.shape} dtype={arr.dtype} ({arr.nbytes/1e6:.1f} MB)")

    skip = set(big_arrays.keys()) | {"bond_a", "bond_b"}
    small = {}
    for k, v in cache.items():
        if k in skip:
            continue
        if isinstance(v, np.ndarray):
            small[k] = v
        elif isinstance(v, (int, float, str, bool)):
            if isinstance(v, str):
                small[f"_str_{k}"] = np.array([v])
            else:
                small[f"_scalar_{k}"] = np.array([v])

    np.savez(mmap_dir / "meta.npz", **small)
    marker.write_text("ok\n")
    if verbose:
        total_mb = sum(a.nbytes for a in big_arrays.values()) / 1e6
        print(f"[prepare-mmap] done: {mmap_dir} ({total_mb:.1f} MB)")
    return str(mmap_dir)


def load_cache_mmap(mmap_dir: str) -> dict:
    """Load cache with big arrays memory-mapped read-only."""
    mmap_dir = Path(mmap_dir)
    cache = {}

    # Mmap big arrays
    for name in (
        "cfg", "bond_hist", "species_counts", "group_keys", "group_ptr",
        "patch_to_species", "patch_to_small", "m_patch", "patch_group_ptr",
        "boundary_orbit_mult", "group_orbit_mult",
    ):
        path = mmap_dir / f"{name}.npy"
        if path.exists():
            cache[name] = np.load(path, mmap_mode="r")

    # Load small arrays / scalars
    meta = np.load(mmap_dir / "meta.npz", allow_pickle=False)
    for k in meta.files:
        if k.startswith("_scalar_"):
            real_key = k[len("_scalar_"):]
            val = meta[k][0]
            if isinstance(val, (np.integer,)):
                cache[real_key] = int(val)
            elif isinstance(val, (np.floating,)):
                cache[real_key] = float(val)
            elif isinstance(val, (np.bool_,)):
                cache[real_key] = bool(val)
            else:
                cache[real_key] = val
        elif k.startswith("_str_"):
            real_key = k[len("_str_"):]
            cache[real_key] = str(meta[k][0])
        else:
            cache[k] = np.array(meta[k])

    return cache


# ===================================================================
# Pre-allocated workspace for cubes_from_cache
# ===================================================================

class CubeWorkspace:
    """Pre-allocated buffers reused across cubes_from_cache calls.

    Thermodynamic convention used below:
      - cube_to_species[g, :] is the Boltzmann average of species counts inside class g.
      - logZ_internal[g] is the Option-B / quotiented internal partition log-weight.
        If the cache stores the full rotational orbit, the pure global-rotation
        orbit factor is divided out by subtracting log(boundary_orbit_mult).
      - class_free_energy[g] = -logZ_internal[g].  This is the object that must
        enter the reduced free energy as e_linear.  Do not use <E> as e_linear.
    """

    def __init__(self, cache: dict, *, use_boundary_quotient: bool = True):
        n_groups = int(cache["n_groups"])
        n_species = int(cache["n_species"])

        self.n_groups = n_groups
        self.n_species = n_species
        self.cube_to_species = np.empty((n_groups, n_species), dtype=np.float64)
        self.class_free_energy = np.empty(n_groups, dtype=np.float64)
        self.logZ_internal = np.empty(n_groups, dtype=np.float64)
        # Occupancy-resolved internal partition coefficients.  Entry (g,n) is
        # log sum_{c in g, Nocc(c)=n} exp[-E_c], including the same class-level
        # rotational quotient as logZ_internal.  These coefficients allow the
        # composition multiplier nu to enter the internal class partition
        # function exactly during the reduced solve.
        self.logZ_by_nocc = np.full((n_groups, 9), -np.inf, dtype=np.float64)
        self.avg_Eeff = np.empty(n_groups, dtype=np.float64)
        self.cube_configs = np.empty((n_groups, 8), dtype=np.int32)

        # Undirected face-pattern basis. Face type is only the 4-slot pattern,
        # not direction x pattern. The face-face registry average is rebuilt
        # from J at every state point.
        slots = np.asarray(cache["small_face_slots"], dtype=np.int64)
        n_ft = len(slots)
        self.slots = slots
        self.n_ft = n_ft
        self.eps_small = np.empty((n_ft, n_ft), dtype=np.float64)
        self.boltz_face = np.empty((n_ft, n_ft), dtype=np.float64)
        # Face-contact registries for two opposing 2x2 faces.
        # Slots are row-major: [0,1,2,3] = [[0,1],[2,3]].
        # When two faces touch, the second face is viewed from the opposite side,
        # so the 0-degree contact registry is left-right mirrored:
        #   A0-B1, A1-B0, A2-B3, A3-B2.
        # The remaining registries are the three in-plane rotations of that
        # mirrored contact face.
        self.perms4 = np.asarray([
            [1, 0, 3, 2],  # 0 deg contact registry
            [2, 1, 0, 3],  # 90 deg contact registry
            [3, 2, 1, 0],  # 180 deg contact registry
            [0, 3, 2, 1],  # 270 deg contact registry
        ], dtype=np.int64)

        # Persistent boundary metadata. These can be large for orbit-averaged
        # boundary classes, so keep the mmap-backed arrays and avoid per-worker
        # float64/int64 copies. The Numba kernels cast indices to int internally.
        self.m_patch = np.asarray(cache["m_patch"])
        self.patch_to_species = np.asarray(cache["patch_to_species"])
        self.patch_to_small = np.asarray(cache["patch_to_small"])

        # Option-B / quotiented global-rotation convention:
        #   A canonical cube class is a renormalized species modulo global rotation.
        #   If cfg_includes_full_orbit is true, the config rows already contain all
        #   rotated representatives.  Their raw log-sum-exp therefore contains a
        #   pure global-orbit entropy log(boundary_orbit_mult).  We subtract that
        #   factor so low-symmetry boundary structures are not artificially favored
        #   only because they have more lab-frame orientations.
        #
        #   Normalized observables are still averaged over the full stored cfg list.
        #   The subtraction affects only the class partition weight, not <n_s> or
        #   <Eeff>.
        #
        #   If cfg_includes_full_orbit is false, the stored cfg list is already a
        #   representative-level quotient, so boundary_orbit_mult is not added.
        cfg_includes_full_orbit = bool(cache.get("cfg_includes_full_orbit", False))
        group_mult = np.asarray(cache.get("group_orbit_mult", np.ones(n_groups)), dtype=np.float64).copy()
        boundary_mult = np.asarray(cache.get("boundary_orbit_mult", np.ones(n_groups)), dtype=np.float64).copy()

        # Diagnostic switch:
        #   use_boundary_quotient=True  : original Option-B convention, subtracting
        #                                log(boundary_orbit_mult) when cfg includes
        #                                the full boundary orbit.
        #   use_boundary_quotient=False : no boundary-orbit division in logZ_internal.
        # Mixing entropy still counts each canonical cube class once because
        # log_mult_np is zero in evaluate_state_point.
        self.use_boundary_quotient = bool(use_boundary_quotient)
        self.cfg_includes_full_orbit = cfg_includes_full_orbit
        self.boundary_orbit_mult_stats = (
            float(np.nanmin(boundary_mult)),
            float(np.nanmax(boundary_mult)),
            float(np.nanmean(boundary_mult)),
        )
        if cfg_includes_full_orbit and self.use_boundary_quotient:
            self.extra_log_mult = np.log(np.maximum(group_mult, 1e-300)) - np.log(np.maximum(boundary_mult, 1e-300))
        else:
            self.extra_log_mult = np.log(np.maximum(group_mult, 1e-300))


@_njit(cache=True)
def _boltzmann_avg_groups_logz(
    Eeff, sc, cfg, gptr, g_start, g_end, lo, extra_log_mult,
    out_cts, out_free_energy, out_logZ, out_logZ_by_nocc,
    out_avg_Eeff, out_cfg,
):
    """Per-group Boltzmann reduction.

    For each canonical cube class g, compute
        logZ_g = log sum_c exp[-(E_c - mu.n_c)] + extra_log_mult[g],
        F_g = -logZ_g,
        <n_s>_g and <Eeff>_g using normalized Boltzmann weights.

    In Option B, extra_log_mult subtracts log(boundary_orbit_mult) when the
    stored cfg list includes the full rotational orbit.  This quotients out
    pure global orientation entropy from the class free energy while leaving
    normalized Boltzmann averages over the stored microstates unchanged.
    """
    n_species = out_cts.shape[1]
    n_occ_bins = out_logZ_by_nocc.shape[1]
    for g in range(g_start, g_end):
        sl_lo = gptr[g] - lo
        sl_hi = gptr[g + 1] - lo

        # Build occupancy-resolved log partition coefficients using a stable
        # online log-add-exp.  The final class-level quotient is a constant for
        # the class and is therefore added to every finite occupancy sector.
        for n in range(n_occ_bins):
            out_logZ_by_nocc[g, n] = -np.inf
        for r in range(sl_lo, sl_hi):
            nocc = 0
            for ss in range(n_species - 1):
                nocc += int(sc[r, ss])
            if nocc < 0 or nocc >= n_occ_bins:
                continue
            x = -Eeff[r]
            old_log = out_logZ_by_nocc[g, nocc]
            if not np.isfinite(old_log):
                out_logZ_by_nocc[g, nocc] = x
            elif x > old_log:
                out_logZ_by_nocc[g, nocc] = x + np.log1p(np.exp(old_log - x))
            else:
                out_logZ_by_nocc[g, nocc] = old_log + np.log1p(np.exp(x - old_log))
        for n in range(n_occ_bins):
            if np.isfinite(out_logZ_by_nocc[g, n]):
                out_logZ_by_nocc[g, n] += extra_log_mult[g]

        Emin = Eeff[sl_lo]
        best_r = sl_lo
        for r in range(sl_lo + 1, sl_hi):
            if Eeff[r] < Emin:
                Emin = Eeff[r]
                best_r = r

        Z_shifted = 0.0
        for r in range(sl_lo, sl_hi):
            Z_shifted += np.exp(-(Eeff[r] - Emin))
        if Z_shifted < 1e-300:
            Z_shifted = 1e-300

        logZ = -Emin + np.log(Z_shifted) + extra_log_mult[g]
        out_logZ[g] = logZ
        out_free_energy[g] = -logZ

        invZ = 1.0 / Z_shifted
        for s in range(n_species):
            out_cts[g, s] = 0.0
        out_avg_Eeff[g] = 0.0

        for r in range(sl_lo, sl_hi):
            w = np.exp(-(Eeff[r] - Emin)) * invZ
            for s in range(n_species):
                out_cts[g, s] += w * sc[r, s]
            out_avg_Eeff[g] += w * Eeff[r]

        for c in range(8):
            out_cfg[g, c] = cfg[best_r, c]


def cubes_from_cache_fast(cache: dict, J: np.ndarray, mu: np.ndarray, ws: CubeWorkspace, *, registry_mode: str = "boltzmann"):
    """Memory-efficient cube-class reduction using bond histograms.

    Returns class_free_energy = -logZ_internal, not <E>.  This is the
    thermodynamically consistent reduced energy for the solver.
    """
    J = np.asarray(J, dtype=np.float64)
    n_patch_types = int(cache["n_patch_types"])
    n_groups = ws.n_groups

    mu_vec = np.asarray(mu, dtype=np.float64)
    J_flat = J.ravel()

    group_ptr = np.asarray(cache["group_ptr"], dtype=np.int64)
    bond_hist = cache["bond_hist"]
    species_counts = cache["species_counts"]
    cfg_arr = cache["cfg"]

    CHUNK = 512
    for g_start in range(0, n_groups, CHUNK):
        g_end = min(n_groups, g_start + CHUNK)
        lo = int(group_ptr[g_start])
        hi = int(group_ptr[g_end])
        if hi <= lo:
            continue

        chunk_size = hi - lo

        # Use float64 here.  The chunk is modest; logZ/free-energy curvature is
        # more sensitive to small systematic energy errors than the old <E> path.
        E_chunk = np.dot(
            np.asarray(bond_hist[lo:hi], dtype=np.float64),
            J_flat,
        )

        cfg_chunk = np.asarray(cfg_arr[lo:hi], dtype=np.int32)
        mu_sum = np.zeros(chunk_size, dtype=np.float64)
        for col in range(8):
            mu_sum += mu_vec[cfg_chunk[:, col].astype(np.intp)]

        Eeff_chunk = E_chunk - mu_sum
        sc_chunk = np.asarray(species_counts[lo:hi], dtype=np.float64)

        _boltzmann_avg_groups_logz(
            Eeff_chunk, sc_chunk, cfg_chunk, group_ptr,
            g_start, g_end, lo, ws.extra_log_mult,
            ws.cube_to_species, ws.class_free_energy, ws.logZ_internal,
            ws.logZ_by_nocc, ws.avg_Eeff, ws.cube_configs,
        )

    # Face-face effective energy on the undirected M**4 basis.
    # The four registries are the four in-plane contact registries of the second
    # 2x2 face, including the required mirror for opposing faces at 0 degrees:
    #   E0   = J[a0,b1] + J[a1,b0] + J[a2,b3] + J[a3,b2]
    #   E90  = J[a0,b3] + J[a1,b1] + J[a2,b2] + J[a3,b0]
    #   E180 = J[a0,b2] + J[a1,b3] + J[a2,b0] + J[a3,b1]
    #   E270 = J[a0,b0] + J[a1,b2] + J[a2,b1] + J[a3,b3]
    #
    # registry_mode="boltzmann": eps_small = -log(mean_r exp[-E_r])
    # registry_mode="min":       eps_small = min_r E_r
    #
    # The state-point routine then uses delta = (exp(-eps_small) - 1) / factor.
    reg_mode = str(registry_mode).strip().lower().replace("-", "_")
    if reg_mode in ("boltzmann", "boltz", "average", "avg", "boltzmann_average", "registry_average"):
        ws.boltz_face[:] = 0.0
        for r in range(4):
            bp = ws.slots[:, ws.perms4[r]]
            Ereg = np.zeros_like(ws.boltz_face)
            for k in range(4):
                Ereg += J[ws.slots[:, k][:, None], bp[:, k][None, :]]
            ws.boltz_face += np.exp(-Ereg)
        ws.boltz_face *= 0.25
        ws.eps_small[:] = -np.log(np.maximum(ws.boltz_face, 1e-300))
    elif reg_mode in ("min", "minimum", "min_energy", "best", "best_registry", "most_compatible"):
        ws.eps_small[:] = np.inf
        for r in range(4):
            bp = ws.slots[:, ws.perms4[r]]
            Ereg = np.zeros_like(ws.eps_small)
            for k in range(4):
                Ereg += J[ws.slots[:, k][:, None], bp[:, k][None, :]]
            np.minimum(ws.eps_small, Ereg, out=ws.eps_small)
    else:
        raise ValueError(
            f"Unknown registry_mode={registry_mode!r}. Use 'boltzmann' or 'min'."
        )

    return (
        ws.eps_small,
        ws.class_free_energy,
        ws.logZ_internal,
        ws.logZ_by_nocc,
        ws.avg_Eeff,
        ws.cube_to_species,
        ws.m_patch,
        ws.patch_to_species,
        ws.patch_to_small,
        ws.cube_configs,
    )


# ===================================================================
# Solver: Broyden continuation with finite-difference Jacobian rebuilds
# ===================================================================

def _ts():
    import datetime as _dt
    return _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log(msg):
    print(f"[{_ts()}] {msg}", flush=True)


def contiguous_shard_bounds(n_items, shard_id, n_shards):
    q, r = divmod(n_items, n_shards)
    start = shard_id * q + min(shard_id, r)
    stop = start + q + (1 if shard_id < r else 0)
    return start, stop


def grid_from_args(args):
    eps_as = np.linspace(args.eps_a_min, args.eps_a_max, args.n_eps_a)
    eps_cs = np.linspace(args.eps_c_min, args.eps_c_max, args.n_eps_c)
    return eps_as.astype(np.float64), eps_cs.astype(np.float64)


def loess_derivs(x, y, *, bandwidth=0.25, order=2):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n = x.shape[0]
    f0 = np.zeros(n, dtype=np.float64)
    f1 = np.zeros(n, dtype=np.float64)
    f2 = np.zeros(n, dtype=np.float64)
    h = bandwidth * (x.max() - x.min())
    if h <= 0.0:
        h = 1.0
    for i in range(n):
        d = np.abs(x - x[i])
        w = np.clip(1.0 - (d / h) ** 3, 0.0, None) ** 3
        dx = x - x[i]
        cols = [np.ones_like(dx)]
        if order >= 1:
            cols.append(dx)
        if order >= 2:
            cols.append(dx ** 2)
        X = np.stack(cols, axis=1)
        XTW = X.T * w[None, :]
        M = XTW @ X
        ridge = 1e-12 * np.eye(M.shape[0], dtype=np.float64)
        beta = np.linalg.solve(M + ridge, XTW @ y)
        f0[i] = beta[0]
        f1[i] = beta[1] if order >= 1 else np.nan
        f2[i] = 2.0 * beta[2] if order >= 2 else np.nan
    return f0, f1, f2


def zero_crossings_linear(phis, y, tol=1e-6):
    z = []
    for i in range(len(phis) - 1):
        y1, y2 = y[i], y[i + 1]
        if abs(y1) < tol:
            z.append(float(phis[i]))
            continue
        if y1 * y2 < 0:
            t = -y1 / (y2 - y1)
            z.append(float(phis[i] + t * (phis[i + 1] - phis[i])))
    if abs(y[-1]) < tol:
        z.append(float(phis[-1]))
    return z


def _lower_convex_envelope_indices(phi, f):
    """Return indices on the lower convex envelope for sorted 1D data.

    ConvexHull(vertices) contains both the upper and lower hull.  Sorting those
    vertices by index is not enough: upper-hull vertices can sit between two
    lower-hull endpoints and hide the common-tangent segment.  For binodals we
    need the lower convex envelope / convex minorant only.

    For increasing phi, the lower convex envelope has nondecreasing slopes.
    The monotone-chain update below removes the middle point whenever the new
    slope would make the envelope locally concave.
    """
    phi = np.asarray(phi, dtype=np.float64)
    f = np.asarray(f, dtype=np.float64)
    finite = np.isfinite(phi) & np.isfinite(f)
    idx = np.nonzero(finite)[0]
    if idx.size == 0:
        return np.array([], dtype=np.int64)

    # The caller normally provides a sorted phi grid, but keep this safe.
    idx = idx[np.argsort(phi[idx], kind="mergesort")]

    # Drop duplicate phi values, keeping the lowest f because the lower envelope
    # cannot use the higher duplicate at the same composition.
    dedup = []
    for ii in idx:
        if not dedup or phi[ii] > phi[dedup[-1]]:
            dedup.append(int(ii))
        elif f[ii] < f[dedup[-1]]:
            dedup[-1] = int(ii)
    idx = np.asarray(dedup, dtype=np.int64)
    if idx.size <= 2:
        return idx

    hull = []
    for ii in idx:
        hull.append(int(ii))
        while len(hull) >= 3:
            i, j, k = hull[-3], hull[-2], hull[-1]
            s1 = (f[j] - f[i]) / max(phi[j] - phi[i], 1e-300)
            s2 = (f[k] - f[j]) / max(phi[k] - phi[j], 1e-300)
            # Lower convex envelope requires increasing slopes.  If s2 <= s1,
            # the middle point j lies above the convex minorant chord and must
            # be removed.  A tiny tolerance prevents numerical chatter.
            if s2 <= s1 + 1e-13:
                hull.pop(-2)
            else:
                break
    return np.asarray(hull, dtype=np.int64)


def extract_binodals_from_convex_envelope(phi, f, *, coexist_tol=1e-5, min_gap_points=6):
    """Extract common-tangent/binodal candidates from the lower convex envelope.

    A binodal segment is a lower-envelope chord for which the original free
    energy sits above the chord over an interior composition interval.  The
    barrier is max[f(phi) - chord(phi)] inside that interval.
    """
    phi = np.asarray(phi, dtype=np.float64)
    f = np.asarray(f, dtype=np.float64)
    if phi.size != f.size or phi.size < 2:
        return {"phi": phi, "f": f, "hull_idx": [], "segments": []}

    hull_idx = _lower_convex_envelope_indices(phi, f)
    out = []
    for k in range(len(hull_idx) - 1):
        i, j = int(hull_idx[k]), int(hull_idx[k + 1])
        if j < i:
            i, j = j, i
        if j - i < min_gap_points:
            continue
        phi1, phi2 = float(phi[i]), float(phi[j])
        if not np.isfinite(phi1) or not np.isfinite(phi2) or phi2 <= phi1:
            continue
        f1, f2 = float(f[i]), float(f[j])
        mu_line = (f2 - f1) / (phi2 - phi1)
        line = f1 + mu_line * (phi - phi1)
        diff = f - line
        mask_inner = (phi >= phi1) & (phi <= phi2) & np.isfinite(diff)
        if not np.any(mask_inner):
            continue
        barrier = float(np.max(diff[mask_inner]))
        below_min = float(np.min(diff[mask_inner]))
        # Numerical lower-envelope construction can produce tiny negative
        # below_min from floating noise.  Reject only if the chord clearly lies
        # above the sampled free-energy curve.
        if below_min < -1e-7:
            continue
        if barrier < coexist_tol:
            continue
        out.append({
            "phi1": phi1, "phi2": phi2,
            "mu": float(mu_line), "barrier": barrier,
            "below_min": below_min, "n_points": int(j - i),
        })
    return {"phi": phi, "f": f, "hull_idx": hull_idx.tolist(), "segments": out}


def best_binodal_segment(result, *, prefer="largest_barrier"):
    segs = result.get("segments", [])
    if not segs:
        return None
    if prefer == "widest":
        key = lambda d: (d["phi2"] - d["phi1"], d["barrier"])
    elif prefer == "most_points":
        key = lambda d: (d["n_points"], d["barrier"])
    else:
        key = lambda d: (d["barrier"], d["phi2"] - d["phi1"])
    return max(segs, key=key)


# ---------------------------------------------------------------------------
# Numba solver kernels
# ---------------------------------------------------------------------------

@_njit(cache=True)
def _apply_g_from_c(c, pts, pss, m_patch, P):
    g = np.zeros(P, dtype=np.float64)
    for a in range(len(pts)):
        gi = int(pts[a])
        si = int(pss[a])
        g[gi] += m_patch[a] * c[si]
    return g


@_njit(cache=True)
def _apply_pr_from_rho(rho, pts, pss, m_patch, Fd):
    pr = np.zeros(Fd, dtype=np.float64)
    for a in range(len(pts)):
        gi = int(pts[a])
        si = int(pss[a])
        pr[si] += m_patch[a] * rho[gi]
    return pr


@_njit(cache=True)
def _eval_residual_reduced_cube_sparse(mu, W, phi, pts, pss, m_patch, delta, A, e, lm):
    Fd = W.shape[0]
    P = A.shape[0]

    u = delta @ W
    X = np.empty(Fd, dtype=np.float64)
    hv = np.empty(Fd, dtype=np.float64)
    for s in range(Fd):
        xs = 1.0 / max(1.0 + u[s], 1e-12)
        xs = max(min(xs, 1.0), 1e-12)
        X[s] = xs
        hv[s] = np.log(max(xs, 1e-300)) - 0.5 * xs + 0.5

    g0 = _apply_g_from_c(hv, pts, pss, m_patch, P)
    logits = np.empty(P, dtype=np.float64)
    mx = -1e300
    for i in range(P):
        v = lm[i] + 8.0 * mu * A[i] - e[i] - g0[i]
        logits[i] = v
        if v > mx:
            mx = v

    rho = np.empty(P, dtype=np.float64)
    Z = 0.0
    for i in range(P):
        ri = np.exp(logits[i] - mx)
        rho[i] = ri
        Z += ri
    invZ = 1.0 / Z
    for i in range(P):
        rho[i] *= invZ

    pr = _apply_pr_from_rho(rho, pts, pss, m_patch, Fd)

    K = np.empty((Fd, Fd), dtype=np.float64)
    for s in range(Fd):
        xs2 = X[s] * X[s]
        for sp in range(Fd):
            K[s, sp] = xs2 * delta[s, sp] * pr[sp]

    dAdX = np.empty(Fd, dtype=np.float64)
    for s in range(Fd):
        dAdX[s] = pr[s] * (1.0 / max(X[s], 1e-12) - 0.5)

    Msys = np.empty((Fd, Fd), dtype=np.float64)
    for s in range(Fd):
        for sp in range(Fd):
            Msys[s, sp] = K[sp, s]
        Msys[s, s] += 1.0 + 1e-12

    alpha = np.linalg.solve(Msys, dAdX)

    beta = np.zeros(Fd, dtype=np.float64)
    for s in range(Fd):
        acc = 0.0
        for sp in range(Fd):
            acc += delta[sp, s] * alpha[sp] * X[sp] * X[sp]
        beta[s] = -acc

    c = np.empty(Fd, dtype=np.float64)
    for s in range(Fd):
        c[s] = hv[s] + beta[s] * X[s]

    g = _apply_g_from_c(c, pts, pss, m_patch, P)
    mx = -1e300
    for i in range(P):
        v = lm[i] + 8.0 * mu * A[i] - e[i] - g[i]
        logits[i] = v
        if v > mx:
            mx = v

    Z = 0.0
    for i in range(P):
        ri = np.exp(logits[i] - mx)
        rho[i] = ri
        Z += ri
    invZ = 1.0 / Z
    for i in range(P):
        rho[i] *= invZ

    pr = _apply_pr_from_rho(rho, pts, pss, m_patch, Fd)

    R = np.empty(1 + Fd, dtype=np.float64)
    phi_rho = 0.0
    for i in range(P):
        phi_rho += A[i] * rho[i]
    R[0] = phi_rho - phi
    for s in range(Fd):
        R[1 + s] = W[s] - X[s] * pr[s]

    A_mix = 0.0
    for i in range(P):
        A_mix += rho[i] * (np.log(max(rho[i], 1e-300)) - lm[i])

    A_lin = 0.0
    for i in range(P):
        A_lin += e[i] * rho[i]

    A_assoc = 0.0
    for s in range(Fd):
        A_assoc += hv[s] * pr[s]

    f = (A_mix + A_lin + A_assoc) / 8.0
    return R, f


@_njit(cache=True)
def _find_initial_mu_cube_sparse(phi_target, W, pts, pss, m_patch, delta, A, e, lm):
    Fd = W.shape[0]
    P = A.shape[0]

    u = delta @ W
    hv = np.empty(Fd, dtype=np.float64)
    for s in range(Fd):
        xs = 1.0 / max(1.0 + u[s], 1e-12)
        xs = max(min(xs, 1.0), 1e-12)
        hv[s] = np.log(max(xs, 1e-300)) - 0.5 * xs + 0.5

    g = _apply_g_from_c(hv, pts, pss, m_patch, P)

    mu_lo = -100.0
    mu_hi = 100.0
    mu = 0.0
    for _ in range(120):
        mx = -1e300
        for i in range(P):
            v = lm[i] + 8.0 * mu * A[i] - e[i] - g[i]
            if v > mx:
                mx = v

        Z = 0.0
        phi_mu = 0.0
        second = 0.0
        for i in range(P):
            ri = np.exp(lm[i] + 8.0 * mu * A[i] - e[i] - g[i] - mx)
            Z += ri
        invZ = 1.0 / Z

        for i in range(P):
            ri = np.exp(lm[i] + 8.0 * mu * A[i] - e[i] - g[i] - mx) * invZ
            phi_mu += A[i] * ri
            second += A[i] * A[i] * ri

        dphi = 8.0 * (second - phi_mu * phi_mu)
        err = phi_mu - phi_target
        if abs(err) < 1e-12:
            break
        if err > 0.0:
            mu_hi = mu
        else:
            mu_lo = mu

        if abs(dphi) > 1e-30:
            mu_new = mu - err / dphi
            if mu_new < mu_lo or mu_new > mu_hi:
                mu_new = 0.5 * (mu_lo + mu_hi)
        else:
            mu_new = 0.5 * (mu_lo + mu_hi)
        mu = mu_new
    return mu


def _build_fd_jacobian(mu, W, phi, pts, pss, m_patch, delta, A, e, lm, R, h_fd=1e-7):
    """Compute full finite-difference Jacobian (expensive — 1+Fd evals).

    Called ONCE at the start of each phi solve or when Broyden stalls.
    """
    Fd = W.shape[0]
    n = 1 + Fd
    Jac = np.empty((n, n), dtype=np.float64)

    Rp, _ = _eval_residual_reduced_cube_sparse(
        mu + h_fd, W, phi, pts, pss, m_patch, delta, A, e, lm)
    Jac[:, 0] = (Rp - R) / h_fd

    for j in range(Fd):
        Wp = W.copy()
        Wp[j] += h_fd
        Rp, _ = _eval_residual_reduced_cube_sparse(
            mu, Wp, phi, pts, pss, m_patch, delta, A, e, lm)
        Jac[:, 1 + j] = (Rp - R) / h_fd

    Jac[np.diag_indices_from(Jac)] += 1e-12
    return Jac


def _solve_single_phi_broyden(
    phi, mu0, W0, pts, pss, m_patch, delta, A, e, lm,
    *, tol=1e-8, max_iter=30, step_cap=10.0, h_fd=1e-7,
    prev_B=None, max_jac_rebuilds=1,
):
    """Solve for a single phi using Broyden's method with rank-1 Jacobian updates.

    The key cost limiter is max_jac_rebuilds: each rebuild costs 1+Fd evals
    (= 348 for Fd=347).  The old code allowed unlimited rebuilds — with 30
    iterations stalling every 3, that's 10 rebuilds × 348 = 3,480 evals per
    phi.  Across 51 phis this dominated total runtime.

    With max_jac_rebuilds=1 per phi, worst case is:
      348 (initial) + 348 (1 rebuild) + 30×4 (line search) = 816 evals/phi
    vs the old 4,068 evals/phi.  5× faster for hard points.
    """
    Fd = W0.shape[0]
    n = 1 + Fd

    mu = float(mu0)
    W = W0.copy()

    R, f = _eval_residual_reduced_cube_sparse(mu, W, phi, pts, pss, m_patch, delta, A, e, lm)
    rn = float(np.linalg.norm(R))

    if rn > 0.1:
        mu = _find_initial_mu_cube_sparse(phi, W, pts, pss, m_patch, delta, A, e, lm)
        R, f = _eval_residual_reduced_cube_sparse(mu, W, phi, pts, pss, m_patch, delta, A, e, lm)
        rn = float(np.linalg.norm(R))

    best_x = np.empty(n, dtype=np.float64)
    best_x[0] = mu
    best_x[1:] = W.copy()
    best_rn = rn
    best_f = float(f) if np.isfinite(f) else np.nan

    if rn < tol:
        return {"x": best_x, "rn": best_rn, "f": best_f, "success": True, "B": prev_B}

    # Build inverse Jacobian B = J^{-1}
    n_jac_builds = 0
    if prev_B is not None:
        B = prev_B.copy()
    else:
        Jac = _build_fd_jacobian(mu, W, phi, pts, pss, m_patch, delta, A, e, lm, R, h_fd)
        try:
            B = np.linalg.inv(Jac)
        except np.linalg.LinAlgError:
            B = np.eye(n, dtype=np.float64)
        n_jac_builds = 1

    stall_count = 0

    for it in range(max_iter):
        # Broyden step: dx = -B @ R
        dx = -(B @ R)

        # Cap step size
        dx_norm = float(np.linalg.norm(dx))
        if dx_norm > step_cap:
            dx *= step_cap / dx_norm

        # Backtracking line search (4 steps max — don't waste evals)
        alpha = 1.0
        R_new = None
        f_new = np.nan
        rn_new = rn
        for _ in range(4):
            mu_try = mu + alpha * dx[0]
            W_try = W + alpha * dx[1:]
            R_try, f_try = _eval_residual_reduced_cube_sparse(
                mu_try, W_try, phi, pts, pss, m_patch, delta, A, e, lm)
            rn_try = float(np.linalg.norm(R_try))
            if rn_try < rn:
                R_new = R_try
                f_new = f_try
                rn_new = rn_try
                break
            alpha *= 0.5

        if R_new is None:
            # Line search failed — stall
            stall_count += 1
            if stall_count >= 3 and n_jac_builds < max_jac_rebuilds:
                # One rebuild allowed — use it
                Jac = _build_fd_jacobian(mu, W, phi, pts, pss, m_patch, delta, A, e, lm, R, h_fd)
                try:
                    B = np.linalg.inv(Jac)
                except np.linalg.LinAlgError:
                    break
                n_jac_builds += 1
                stall_count = 0
            elif stall_count >= 3:
                # Already used our rebuild budget — give up on this phi
                break
            continue

        stall_count = 0

        # Broyden rank-1 inverse Jacobian update (Sherman-Morrison)
        s = alpha * dx                          # actual step taken
        y = R_new - R                           # residual change
        By = B @ y
        sTBy = float(s @ By)
        if abs(sTBy) > 1e-30:
            # B_new = B + (s - B@y) (s^T B) / (s^T B y)
            B += np.outer(s - By, s @ B) / sTBy

        # Accept step
        mu += alpha * dx[0]
        W += alpha * dx[1:]
        R = R_new
        f = f_new
        rn = rn_new

        if rn < best_rn:
            best_x[0] = mu
            best_x[1:] = W.copy()
            best_rn = rn
            best_f = float(f)

        if rn < tol:
            break

    return {
        "x": best_x, "rn": best_rn, "f": best_f,
        "success": best_rn <= tol, "B": B,
    }


def _solve_single_phi_newton(
    phi, mu0, W0, pts, pss, m_patch, delta, A, e, lm,
    *, tol=1e-8, max_iter=30, step_cap=10.0, h_fd=1e-7,
):
    """Full Broyden solver with FD Jacobian every iteration.

    Slow (348 evals/iteration) but reliable — stays on the correct
    solution branch.  Used as the ground-truth solver.
    """
    Fd = W0.shape[0]
    n = 1 + Fd

    mu = float(mu0)
    W = W0.copy()

    R, f = _eval_residual_reduced_cube_sparse(mu, W, phi, pts, pss, m_patch, delta, A, e, lm)
    rn = float(np.linalg.norm(R))
    if rn > 0.1:
        mu = _find_initial_mu_cube_sparse(phi, W, pts, pss, m_patch, delta, A, e, lm)
        R, f = _eval_residual_reduced_cube_sparse(mu, W, phi, pts, pss, m_patch, delta, A, e, lm)
        rn = float(np.linalg.norm(R))

    best_x = np.empty(n, dtype=np.float64)
    best_x[0] = mu
    best_x[1:] = W.copy()
    best_rn = rn
    best_f = float(f) if np.isfinite(f) else np.nan

    for it in range(max_iter):
        if rn < tol:
            break
        Jac = _build_fd_jacobian(mu, W, phi, pts, pss, m_patch, delta, A, e, lm, R, h_fd)
        try:
            dp = np.linalg.solve(Jac, -R)
        except np.linalg.LinAlgError:
            break
        np.clip(dp, -step_cap, step_cap, out=dp)

        alpha = 1.0
        for _ in range(8):
            R_try, f_try = _eval_residual_reduced_cube_sparse(
                mu + alpha * dp[0], W + alpha * dp[1:],
                phi, pts, pss, m_patch, delta, A, e, lm)
            if float(np.linalg.norm(R_try)) < rn:
                break
            alpha *= 0.5

        mu += alpha * dp[0]
        W += alpha * dp[1:]
        R, f = _eval_residual_reduced_cube_sparse(mu, W, phi, pts, pss, m_patch, delta, A, e, lm)
        rn = float(np.linalg.norm(R))
        if rn < best_rn:
            best_x[0] = mu
            best_x[1:] = W.copy()
            best_rn = rn
            best_f = float(f)

    return {"x": best_x, "rn": best_rn, "f": best_f, "success": best_rn <= tol}



@_njit(cache=True)
def _eval_residual_reduced_cube_sparse_field(mu, W, phi, pts, pss, m_patch, delta, A, e, lm, pi_ext):
    """Same reduced SAFT-P residual, with an external cube-state field pi_ext.

    pi_ext is the derivative of the self-consistent Bethe compatibility penalty
    with respect to rho_g.  The stationarity softmax therefore uses
        -e_g - (M c)_g - pi_ext[g].
    The returned f is the base SAFT-P free energy only; the integrated Bethe
    penalty is added outside after the final rho is known.
    """
    Fd = W.shape[0]
    P = A.shape[0]

    u = delta @ W
    X = np.empty(Fd, dtype=np.float64)
    hv = np.empty(Fd, dtype=np.float64)
    for s in range(Fd):
        xs = 1.0 / max(1.0 + u[s], 1e-12)
        xs = max(min(xs, 1.0), 1e-12)
        X[s] = xs
        hv[s] = np.log(max(xs, 1e-300)) - 0.5 * xs + 0.5

    g0 = _apply_g_from_c(hv, pts, pss, m_patch, P)
    logits = np.empty(P, dtype=np.float64)
    mx = -1e300
    for i in range(P):
        v = lm[i] + 8.0 * mu * A[i] - e[i] - g0[i] - pi_ext[i]
        logits[i] = v
        if v > mx:
            mx = v

    rho = np.empty(P, dtype=np.float64)
    Z = 0.0
    for i in range(P):
        ri = np.exp(logits[i] - mx)
        rho[i] = ri
        Z += ri
    invZ = 1.0 / Z
    for i in range(P):
        rho[i] *= invZ

    pr = _apply_pr_from_rho(rho, pts, pss, m_patch, Fd)

    K = np.empty((Fd, Fd), dtype=np.float64)
    for s in range(Fd):
        xs2 = X[s] * X[s]
        for sp in range(Fd):
            K[s, sp] = xs2 * delta[s, sp] * pr[sp]

    dAdX = np.empty(Fd, dtype=np.float64)
    for s in range(Fd):
        dAdX[s] = pr[s] * (1.0 / max(X[s], 1e-12) - 0.5)

    Msys = np.empty((Fd, Fd), dtype=np.float64)
    for s in range(Fd):
        for sp in range(Fd):
            Msys[s, sp] = K[sp, s]
        Msys[s, s] += 1.0 + 1e-12

    alpha = np.linalg.solve(Msys, dAdX)

    beta = np.zeros(Fd, dtype=np.float64)
    for s in range(Fd):
        acc = 0.0
        for sp in range(Fd):
            acc += delta[sp, s] * alpha[sp] * X[sp] * X[sp]
        beta[s] = -acc

    c = np.empty(Fd, dtype=np.float64)
    for s in range(Fd):
        c[s] = hv[s] + beta[s] * X[s]

    g = _apply_g_from_c(c, pts, pss, m_patch, P)
    mx = -1e300
    for i in range(P):
        v = lm[i] + 8.0 * mu * A[i] - e[i] - g[i] - pi_ext[i]
        logits[i] = v
        if v > mx:
            mx = v

    Z = 0.0
    for i in range(P):
        ri = np.exp(logits[i] - mx)
        rho[i] = ri
        Z += ri
    invZ = 1.0 / Z
    for i in range(P):
        rho[i] *= invZ

    pr = _apply_pr_from_rho(rho, pts, pss, m_patch, Fd)

    R = np.empty(1 + Fd, dtype=np.float64)
    phi_rho = 0.0
    for i in range(P):
        phi_rho += A[i] * rho[i]
    R[0] = phi_rho - phi
    for s in range(Fd):
        R[1 + s] = W[s] - X[s] * pr[s]

    A_mix = 0.0
    for i in range(P):
        A_mix += rho[i] * (np.log(max(rho[i], 1e-300)) - lm[i])

    A_lin = 0.0
    for i in range(P):
        A_lin += e[i] * rho[i]

    A_assoc = 0.0
    for s in range(Fd):
        A_assoc += hv[s] * pr[s]

    f = (A_mix + A_lin + A_assoc) / 8.0
    return R, f


@_njit(cache=True)
def _rho_pr_from_solution_cube_sparse_field(mu, W, pts, pss, m_patch, delta, A, e, lm, pi_ext):
    Fd = W.shape[0]
    P = A.shape[0]

    u = delta @ W
    X = np.empty(Fd, dtype=np.float64)
    hv = np.empty(Fd, dtype=np.float64)
    for s in range(Fd):
        xs = 1.0 / max(1.0 + u[s], 1e-12)
        xs = max(min(xs, 1.0), 1e-12)
        X[s] = xs
        hv[s] = np.log(max(xs, 1e-300)) - 0.5 * xs + 0.5

    g0 = _apply_g_from_c(hv, pts, pss, m_patch, P)
    logits = np.empty(P, dtype=np.float64)
    mx = -1e300
    for i in range(P):
        v = lm[i] + 8.0 * mu * A[i] - e[i] - g0[i] - pi_ext[i]
        logits[i] = v
        if v > mx:
            mx = v

    rho = np.empty(P, dtype=np.float64)
    Z = 0.0
    for i in range(P):
        ri = np.exp(logits[i] - mx)
        rho[i] = ri
        Z += ri
    invZ = 1.0 / Z
    for i in range(P):
        rho[i] *= invZ

    pr = _apply_pr_from_rho(rho, pts, pss, m_patch, Fd)

    K = np.empty((Fd, Fd), dtype=np.float64)
    for s in range(Fd):
        xs2 = X[s] * X[s]
        for sp in range(Fd):
            K[s, sp] = xs2 * delta[s, sp] * pr[sp]

    dAdX = np.empty(Fd, dtype=np.float64)
    for s in range(Fd):
        dAdX[s] = pr[s] * (1.0 / max(X[s], 1e-12) - 0.5)

    Msys = np.empty((Fd, Fd), dtype=np.float64)
    for s in range(Fd):
        for sp in range(Fd):
            Msys[s, sp] = K[sp, s]
        Msys[s, s] += 1.0 + 1e-12

    alpha = np.linalg.solve(Msys, dAdX)
    beta = np.zeros(Fd, dtype=np.float64)
    for s in range(Fd):
        acc = 0.0
        for sp in range(Fd):
            acc += delta[sp, s] * alpha[sp] * X[sp] * X[sp]
        beta[s] = -acc

    c = np.empty(Fd, dtype=np.float64)
    for s in range(Fd):
        c[s] = hv[s] + beta[s] * X[s]

    g = _apply_g_from_c(c, pts, pss, m_patch, P)
    mx = -1e300
    for i in range(P):
        v = lm[i] + 8.0 * mu * A[i] - e[i] - g[i] - pi_ext[i]
        logits[i] = v
        if v > mx:
            mx = v

    Z = 0.0
    for i in range(P):
        ri = np.exp(logits[i] - mx)
        rho[i] = ri
        Z += ri
    invZ = 1.0 / Z
    for i in range(P):
        rho[i] *= invZ

    pr = _apply_pr_from_rho(rho, pts, pss, m_patch, Fd)
    return rho, pr


@_njit(cache=True)
def _find_initial_mu_cube_sparse_field(phi_target, W, pts, pss, m_patch, delta, A, e, lm, pi_ext):
    Fd = W.shape[0]
    P = A.shape[0]

    u = delta @ W
    hv = np.empty(Fd, dtype=np.float64)
    for s in range(Fd):
        xs = 1.0 / max(1.0 + u[s], 1e-12)
        xs = max(min(xs, 1.0), 1e-12)
        hv[s] = np.log(max(xs, 1e-300)) - 0.5 * xs + 0.5

    g = _apply_g_from_c(hv, pts, pss, m_patch, P)

    mu_lo = -100.0
    mu_hi = 100.0
    mu = 0.0
    for _ in range(120):
        mx = -1e300
        for i in range(P):
            v = lm[i] + 8.0 * mu * A[i] - e[i] - g[i] - pi_ext[i]
            if v > mx:
                mx = v

        Z = 0.0
        phi_mu = 0.0
        second = 0.0
        for i in range(P):
            ri = np.exp(lm[i] + 8.0 * mu * A[i] - e[i] - g[i] - pi_ext[i] - mx)
            Z += ri
        invZ = 1.0 / Z

        for i in range(P):
            ri = np.exp(lm[i] + 8.0 * mu * A[i] - e[i] - g[i] - pi_ext[i] - mx) * invZ
            phi_mu += A[i] * ri
            second += A[i] * A[i] * ri

        dphi = 8.0 * (second - phi_mu * phi_mu)
        err = phi_mu - phi_target
        if abs(err) < 1e-12:
            break
        if err > 0.0:
            mu_hi = mu
        else:
            mu_lo = mu

        if abs(dphi) > 1e-30:
            mu_new = mu - err / dphi
            if mu_new < mu_lo or mu_new > mu_hi:
                mu_new = 0.5 * (mu_lo + mu_hi)
        else:
            mu_new = 0.5 * (mu_lo + mu_hi)
        mu = mu_new
    return mu


def _build_fd_jacobian_field(mu, W, phi, pts, pss, m_patch, delta, A, e, lm, pi_ext, R, h_fd=1e-7):
    Fd = W.shape[0]
    n = 1 + Fd
    Jac = np.empty((n, n), dtype=np.float64)

    Rp, _ = _eval_residual_reduced_cube_sparse_field(mu + h_fd, W, phi, pts, pss, m_patch, delta, A, e, lm, pi_ext)
    Jac[:, 0] = (Rp - R) / h_fd

    for j in range(Fd):
        Wp = W.copy()
        Wp[j] += h_fd
        Rp, _ = _eval_residual_reduced_cube_sparse_field(mu, Wp, phi, pts, pss, m_patch, delta, A, e, lm, pi_ext)
        Jac[:, 1 + j] = (Rp - R) / h_fd

    Jac[np.diag_indices_from(Jac)] += 1e-12
    return Jac


def _solve_single_phi_newton_field(
    phi, mu0, W0, pts, pss, m_patch, delta, A, e, lm, pi_ext,
    *, tol=1e-8, max_iter=30, step_cap=10.0, h_fd=1e-7,
):
    Fd = W0.shape[0]
    n = 1 + Fd
    mu = float(mu0)
    W = W0.copy()

    R, f = _eval_residual_reduced_cube_sparse_field(mu, W, phi, pts, pss, m_patch, delta, A, e, lm, pi_ext)
    rn = float(np.linalg.norm(R))
    if rn > 0.1:
        mu = _find_initial_mu_cube_sparse_field(phi, W, pts, pss, m_patch, delta, A, e, lm, pi_ext)
        R, f = _eval_residual_reduced_cube_sparse_field(mu, W, phi, pts, pss, m_patch, delta, A, e, lm, pi_ext)
        rn = float(np.linalg.norm(R))

    best_x = np.empty(n, dtype=np.float64)
    best_x[0] = mu
    best_x[1:] = W.copy()
    best_rn = rn
    best_f = float(f) if np.isfinite(f) else np.nan

    for _it in range(max_iter):
        if rn < tol:
            break
        Jac = _build_fd_jacobian_field(mu, W, phi, pts, pss, m_patch, delta, A, e, lm, pi_ext, R, h_fd)
        try:
            dp = np.linalg.solve(Jac, -R)
        except np.linalg.LinAlgError:
            break
        np.clip(dp, -step_cap, step_cap, out=dp)

        alpha = 1.0
        accepted = False
        for _ls in range(8):
            R_try, f_try = _eval_residual_reduced_cube_sparse_field(
                mu + alpha * dp[0], W + alpha * dp[1:],
                phi, pts, pss, m_patch, delta, A, e, lm, pi_ext)
            if float(np.linalg.norm(R_try)) < rn:
                accepted = True
                break
            alpha *= 0.5
        if not accepted:
            break

        mu += alpha * dp[0]
        W += alpha * dp[1:]
        R, f = _eval_residual_reduced_cube_sparse_field(mu, W, phi, pts, pss, m_patch, delta, A, e, lm, pi_ext)
        rn = float(np.linalg.norm(R))
        if rn < best_rn:
            best_x[0] = mu
            best_x[1:] = W.copy()
            best_rn = rn
            best_f = float(f)

    return {"x": best_x, "rn": best_rn, "f": best_f, "success": best_rn <= tol}





def _solve_single_phi_broyden_field(
    phi, mu0, W0, pts, pss, m_patch, delta, A, e, lm, pi_ext,
    *, tol=1e-8, max_iter=30, step_cap=10.0, h_fd=1e-7,
    prev_B=None, max_jac_rebuilds=0,
):
    """Fast field solver for self-consistent Bethe/state-association runs.

    This is the field analogue of _solve_single_phi_broyden.  It builds at most
    one finite-difference inverse Jacobian per call, then uses inverse-Broyden
    rank-1 updates.  The old field solver rebuilt the full FD Jacobian at every
    Newton iteration; for directed_face_state with ~486 W variables that is the
    main runtime killer.
    """
    Fd = W0.shape[0]
    n = 1 + Fd
    mu = float(mu0)
    W = W0.copy()

    R, f = _eval_residual_reduced_cube_sparse_field(mu, W, phi, pts, pss, m_patch, delta, A, e, lm, pi_ext)
    rn = float(np.linalg.norm(R))
    if rn > 0.1:
        mu = _find_initial_mu_cube_sparse_field(phi, W, pts, pss, m_patch, delta, A, e, lm, pi_ext)
        R, f = _eval_residual_reduced_cube_sparse_field(mu, W, phi, pts, pss, m_patch, delta, A, e, lm, pi_ext)
        rn = float(np.linalg.norm(R))

    best_x = np.empty(n, dtype=np.float64)
    best_x[0] = mu
    best_x[1:] = W.copy()
    best_rn = rn
    best_f = float(f) if np.isfinite(f) else np.nan

    if rn < tol:
        return {"x": best_x, "rn": best_rn, "f": best_f, "success": True, "B": prev_B}

    n_jac_builds = 0
    if prev_B is not None:
        B = prev_B.copy()
    else:
        Jac = _build_fd_jacobian_field(mu, W, phi, pts, pss, m_patch, delta, A, e, lm, pi_ext, R, h_fd)
        try:
            B = np.linalg.inv(Jac)
        except np.linalg.LinAlgError:
            B = np.eye(n, dtype=np.float64)
        n_jac_builds = 1

    stall_count = 0
    for _it in range(max_iter):
        dx = -(B @ R)
        dx_norm = float(np.linalg.norm(dx))
        if dx_norm > step_cap:
            dx *= step_cap / dx_norm

        alpha = 1.0
        R_new = None
        f_new = np.nan
        rn_new = rn
        for _ls in range(4):
            R_try, f_try = _eval_residual_reduced_cube_sparse_field(
                mu + alpha * dx[0], W + alpha * dx[1:],
                phi, pts, pss, m_patch, delta, A, e, lm, pi_ext)
            rn_try = float(np.linalg.norm(R_try))
            if rn_try < rn:
                R_new = R_try
                f_new = f_try
                rn_new = rn_try
                break
            alpha *= 0.5

        if R_new is None:
            stall_count += 1
            if stall_count >= 3 and n_jac_builds < max_jac_rebuilds:
                Jac = _build_fd_jacobian_field(mu, W, phi, pts, pss, m_patch, delta, A, e, lm, pi_ext, R, h_fd)
                try:
                    B = np.linalg.inv(Jac)
                except np.linalg.LinAlgError:
                    break
                n_jac_builds += 1
                stall_count = 0
            elif stall_count >= 3:
                break
            continue

        stall_count = 0
        s_step = alpha * dx
        y = R_new - R
        By = B @ y
        denom = float(s_step @ By)
        if abs(denom) > 1e-30:
            B += np.outer(s_step - By, s_step @ B) / denom

        mu += alpha * dx[0]
        W += alpha * dx[1:]
        R = R_new
        f = f_new
        rn = rn_new

        if rn < best_rn:
            best_x[0] = mu
            best_x[1:] = W.copy()
            best_rn = rn
            best_f = float(f)
        if rn < tol:
            break

    return {"x": best_x, "rn": best_rn, "f": best_f, "success": best_rn <= tol, "B": B}

def filter_feasible_phi_grid(phi_grid, A_row, *, atol=1e-12):
    """Return only phi values inside the actually reachable A range.

    The reduced density variable is A_row = occupied sites per cube class / 8.
    A target phi outside [min(A_row), max(A_row)] is infeasible: no probability
    vector over cube classes can realize it.  These values must not be sent to
    the nonlinear solver because they can create misleading residuals, bad
    continuation states, and fake diagnostics.
    """
    phi_grid = np.asarray(phi_grid, dtype=np.float64)
    A_row = np.asarray(A_row, dtype=np.float64)
    phi_lo = float(np.nanmin(A_row))
    phi_hi = float(np.nanmax(A_row))
    if not np.isfinite(phi_lo) or not np.isfinite(phi_hi):
        raise ValueError("A_row has non-finite min/max; cannot determine feasible phi interval.")
    if phi_hi < phi_lo:
        phi_lo, phi_hi = phi_hi, phi_lo
    mask = (phi_grid >= phi_lo - float(atol)) & (phi_grid <= phi_hi + float(atol))
    return phi_grid[mask], mask, phi_lo, phi_hi



# ---------------------------------------------------------------------------
# Optional Bethe / boundary-state compatibility correction
# ---------------------------------------------------------------------------

@_njit(cache=True)
def _rho_pr_from_solution_cube_sparse(mu, W, pts, pss, m_patch, delta, A, e, lm):
    """Return rho_g and face population pr_s at a converged (mu,W).

    This mirrors the final fixed-point evaluation in
    _eval_residual_reduced_cube_sparse, but returns the cube-class and face-type
    marginals. Bethe corrections use rho_g; the older face-marginal diagnostic
    also uses pr_s.
    """
    Fd = W.shape[0]
    P = A.shape[0]

    u = delta @ W
    X = np.empty(Fd, dtype=np.float64)
    hv = np.empty(Fd, dtype=np.float64)
    for s in range(Fd):
        xs = 1.0 / max(1.0 + u[s], 1e-12)
        xs = max(min(xs, 1.0), 1e-12)
        X[s] = xs
        hv[s] = np.log(max(xs, 1e-300)) - 0.5 * xs + 0.5

    g0 = _apply_g_from_c(hv, pts, pss, m_patch, P)
    logits = np.empty(P, dtype=np.float64)
    mx = -1e300
    for i in range(P):
        v = lm[i] + 8.0 * mu * A[i] - e[i] - g0[i]
        logits[i] = v
        if v > mx:
            mx = v

    rho = np.empty(P, dtype=np.float64)
    Z = 0.0
    for i in range(P):
        ri = np.exp(logits[i] - mx)
        rho[i] = ri
        Z += ri
    invZ = 1.0 / Z
    for i in range(P):
        rho[i] *= invZ

    pr = _apply_pr_from_rho(rho, pts, pss, m_patch, Fd)

    K = np.empty((Fd, Fd), dtype=np.float64)
    for s in range(Fd):
        xs2 = X[s] * X[s]
        for sp in range(Fd):
            K[s, sp] = xs2 * delta[s, sp] * pr[sp]

    dAdX = np.empty(Fd, dtype=np.float64)
    for s in range(Fd):
        dAdX[s] = pr[s] * (1.0 / max(X[s], 1e-12) - 0.5)

    Msys = np.empty((Fd, Fd), dtype=np.float64)
    for s in range(Fd):
        for sp in range(Fd):
            Msys[s, sp] = K[sp, s]
        Msys[s, s] += 1.0 + 1e-12

    alpha = np.linalg.solve(Msys, dAdX)
    beta = np.zeros(Fd, dtype=np.float64)
    for s in range(Fd):
        acc = 0.0
        for sp in range(Fd):
            acc += delta[sp, s] * alpha[sp] * X[sp] * X[sp]
        beta[s] = -acc

    c = np.empty(Fd, dtype=np.float64)
    for s in range(Fd):
        c[s] = hv[s] + beta[s] * X[s]

    g = _apply_g_from_c(c, pts, pss, m_patch, P)
    mx = -1e300
    for i in range(P):
        v = lm[i] + 8.0 * mu * A[i] - e[i] - g[i]
        logits[i] = v
        if v > mx:
            mx = v

    Z = 0.0
    for i in range(P):
        ri = np.exp(logits[i] - mx)
        rho[i] = ri
        Z += ri
    invZ = 1.0 / Z
    for i in range(P):
        rho[i] *= invZ

    pr = _apply_pr_from_rho(rho, pts, pss, m_patch, Fd)
    return rho, pr


def _positive_opposite_face_pairs():
    """Return the three undirected positive cube-face pairs.

    The representative builder convention is [N,E,S,W,T,B].  If the imported
    builder exposes OPPOSITE_FACE, use it; otherwise use the standard fallback.
    """
    try:
        opp = OPPOSITE_FACE
        pairs = []
        for d in range(6):
            od = int(opp[d])
            if d < od:
                pairs.append((d, od))
        if len(pairs) == 3:
            return pairs
    except Exception:
        pass
    return [(0, 2), (1, 3), (4, 5)]


def _build_face_allowed_graph(eps_small, *, threshold=1e-12):
    """Legacy energetic support: allowed if exp(-eps_face)-1 > threshold."""
    eps_small = np.asarray(eps_small, dtype=np.float64)
    excess = np.exp(-eps_small) - 1.0
    allowed = excess > float(threshold)
    if not np.any(allowed):
        allowed = np.ones_like(allowed, dtype=bool)
    allowed = np.logical_or(allowed, allowed.T)
    return np.asarray(allowed, dtype=bool)


def _build_face_allowed_graph_from_slots(slots, perms4, *, compat_mode="slot_exact", eps_small=None, threshold=1e-12):
    """Build a hard, non-energetic face compatibility graph.

    compat_mode:
      - slot_exact: two 2x2 faces are compatible if one mirrored/rotated contact
        registry gives exact slot equality. This is the strict boundary matching
        test and is the default for boundary Bethe.
      - slot_presence: same, but compares occupied/unoccupied slots only.
      - slot_nonconflict: a contact is allowed if no registry pairs two different
        nonzero patch labels; zero is treated as a wildcard.
      - attractive: legacy energetic support from eps_small.
      - all: no hard constraint; correction is zero.
    """
    mode = str(compat_mode).strip().lower().replace("-", "_")
    slots = np.asarray(slots, dtype=np.int64)
    n_ft = int(slots.shape[0])

    if mode in ("attractive", "energy", "energetic", "legacy"):
        if eps_small is None:
            raise ValueError("compat_mode='attractive' requires eps_small")
        return _build_face_allowed_graph(eps_small, threshold=threshold)
    if mode in ("all", "none", "unconstrained"):
        return np.ones((n_ft, n_ft), dtype=bool)

    allowed = np.zeros((n_ft, n_ft), dtype=bool)
    occ = slots > 0
    perms4 = np.asarray(perms4, dtype=np.int64)

    for r in range(perms4.shape[0]):
        bp = slots[:, perms4[r]]
        if mode in ("slot_exact", "exact", "matching", "face_exact"):
            allowed |= np.all(slots[:, None, :] == bp[None, :, :], axis=2)
        elif mode in ("slot_presence", "presence", "occupancy", "binary"):
            bp_occ = occ[:, perms4[r]]
            allowed |= np.all(occ[:, None, :] == bp_occ[None, :, :], axis=2)
        elif mode in ("slot_nonconflict", "nonconflict", "compatible_labels"):
            # Zero means no patch/wildcard.  Nonzero labels must not disagree.
            a = slots[:, None, :]
            b = bp[None, :, :]
            conflict = (a != 0) & (b != 0) & (a != b)
            allowed |= ~np.any(conflict, axis=2)
        else:
            raise ValueError(
                f"Unknown bethe_compat_mode={compat_mode!r}; use slot_exact, "
                "slot_presence, slot_nonconflict, attractive, or all."
            )

    allowed = np.logical_or(allowed, allowed.T)
    return np.asarray(allowed, dtype=bool)


def _face_entropy_penalty_from_marginals(p_row, p_col, allowed):
    """Lower-order compatibility penalty: -log Prob(allowed pair)."""
    p_row = np.maximum(np.asarray(p_row, dtype=np.float64), 0.0)
    p_col = np.maximum(np.asarray(p_col, dtype=np.float64), 0.0)
    sr = float(p_row.sum())
    sc = float(p_col.sum())
    if sr <= 0.0 or sc <= 0.0 or not np.isfinite(sr) or not np.isfinite(sc):
        return np.nan, np.nan, False
    p_row = p_row / sr
    p_col = p_col / sc
    prob = float(np.sum((p_row[:, None] * p_col[None, :]) * allowed))
    prob = max(min(prob, 1.0), 1e-300)
    return -np.log(prob), prob, True


def _bethe_ipf_penalty_from_marginals(p_row, p_col, allowed, *, max_iter=500, tol=1e-10):
    """I-projection KL penalty with fixed row and column marginals.

    Computes
        min_Q sum_ij Q_ij log(Q_ij / (p_i q_j))
    subject to Q_ij=0 where allowed_ij is False, sum_j Q_ij=p_i,
    and sum_i Q_ij=q_j.  This is the Bethe edge compatibility entropy loss.
    """
    p_row = np.maximum(np.asarray(p_row, dtype=np.float64), 0.0)
    p_col = np.maximum(np.asarray(p_col, dtype=np.float64), 0.0)
    sr = float(p_row.sum())
    sc = float(p_col.sum())
    if sr <= 0.0 or sc <= 0.0 or not np.isfinite(sr) or not np.isfinite(sc):
        return np.nan, np.nan, False, 0
    p_row = p_row / sr
    p_col = p_col / sc

    active_r = p_row > 1e-14
    active_c = p_col > 1e-14
    if np.count_nonzero(active_r) == 0 or np.count_nonzero(active_c) == 0:
        return np.nan, np.nan, False, 0

    p0 = p_row[active_r]
    q0 = p_col[active_c]
    A = np.asarray(allowed[np.ix_(active_r, active_c)], dtype=bool)
    if np.any(A.sum(axis=1) == 0) or np.any(A.sum(axis=0) == 0):
        # Infeasible hard support. Return a large finite penalty so scans do not
        # crash, but mark it as not converged.
        prob_allowed = float(np.sum((p0[:, None] * q0[None, :]) * A))
        return 50.0, max(min(prob_allowed, 1.0), 0.0), False, 0

    K = A.astype(np.float64)
    u = np.ones_like(p0)
    v = np.ones_like(q0)
    ok = False
    it_done = 0
    for it in range(int(max_iter)):
        Kv = K @ v
        if np.any(Kv <= 0.0) or not np.all(np.isfinite(Kv)):
            return 50.0, 0.0, False, it
        u = p0 / Kv
        KTu = K.T @ u
        if np.any(KTu <= 0.0) or not np.all(np.isfinite(KTu)):
            return 50.0, 0.0, False, it
        v = q0 / KTu
        if it % 10 == 0 or it == int(max_iter) - 1:
            Q = (u[:, None] * K) * v[None, :]
            row_err = float(np.max(np.abs(Q.sum(axis=1) - p0)))
            col_err = float(np.max(np.abs(Q.sum(axis=0) - q0)))
            if max(row_err, col_err) < float(tol):
                ok = True
                it_done = it + 1
                break
        it_done = it + 1

    Q = (u[:, None] * K) * v[None, :]
    Q = np.maximum(Q, 0.0)
    base = p0[:, None] * q0[None, :]
    mask = Q > 0.0
    kl = float(np.sum(Q[mask] * (np.log(np.maximum(Q[mask], 1e-300)) - np.log(np.maximum(base[mask], 1e-300)))))
    prob_allowed = float(np.sum(base[A]))
    return max(kl, 0.0), max(min(prob_allowed, 1.0), 0.0), bool(ok), int(it_done)


def _bethe_ipf_penalty_from_p(p, allowed, *, max_iter=500, tol=1e-10):
    """Legacy square-marginal wrapper used by face_bethe."""
    return _bethe_ipf_penalty_from_marginals(p, p, allowed, max_iter=max_iter, tol=tol)


def _face_entropy_penalty_from_p(p, allowed):
    """Legacy square-marginal wrapper used by face_entropy."""
    val, prob, ok = _face_entropy_penalty_from_marginals(p, p, allowed)
    return val, prob, ok



def _bethe_ipf_penalty_grad_from_marginals(p_row, p_col, allowed, *, max_iter=500, tol=1e-10):
    """I-projection KL penalty and marginal gradients.

    Solves
        min_Q sum_ij Q_ij log(Q_ij / (p_i q_j))
    subject to Q_ij=0 on forbidden entries and fixed row/column marginals.

    Returns
        kl, allowed_probability, ok, n_iter, grad_row, grad_col

    The gradients are envelope gradients with arbitrary additive constants
    removed by centering against the corresponding marginals.  These constants
    do not affect the rho softmax normalization, but centering improves the
    numerical conditioning of the self-consistent cavity loop.
    """
    p_row_full = np.maximum(np.asarray(p_row, dtype=np.float64), 0.0)
    p_col_full = np.maximum(np.asarray(p_col, dtype=np.float64), 0.0)
    gr_full = np.zeros_like(p_row_full, dtype=np.float64)
    gc_full = np.zeros_like(p_col_full, dtype=np.float64)
    sr = float(p_row_full.sum())
    sc = float(p_col_full.sum())
    if sr <= 0.0 or sc <= 0.0 or not np.isfinite(sr) or not np.isfinite(sc):
        return np.nan, np.nan, False, 0, gr_full, gc_full
    p_row_full = p_row_full / sr
    p_col_full = p_col_full / sc

    active_r = p_row_full > 1e-14
    active_c = p_col_full > 1e-14
    if np.count_nonzero(active_r) == 0 or np.count_nonzero(active_c) == 0:
        return np.nan, np.nan, False, 0, gr_full, gc_full

    p0 = p_row_full[active_r]
    q0 = p_col_full[active_c]
    A = np.asarray(allowed[np.ix_(active_r, active_c)], dtype=bool)
    prob_allowed = float(np.sum((p0[:, None] * q0[None, :]) * A))
    prob_allowed = max(min(prob_allowed, 1.0), 0.0)
    if np.any(A.sum(axis=1) == 0) or np.any(A.sum(axis=0) == 0):
        # Hard infeasible support for the current marginals.  Return a large
        # finite penalty but a zero gradient so the outer solver does not blow up.
        return 50.0, prob_allowed, False, 0, gr_full, gc_full

    K = A.astype(np.float64)
    u = np.ones_like(p0)
    v = np.ones_like(q0)
    ok = False
    it_done = 0
    for it in range(int(max_iter)):
        Kv = K @ v
        if np.any(Kv <= 0.0) or not np.all(np.isfinite(Kv)):
            return 50.0, 0.0, False, it, gr_full, gc_full
        u = p0 / Kv
        KTu = K.T @ u
        if np.any(KTu <= 0.0) or not np.all(np.isfinite(KTu)):
            return 50.0, 0.0, False, it, gr_full, gc_full
        v = q0 / KTu
        it_done = it + 1
        if it % 10 == 0 or it == int(max_iter) - 1:
            Q = (u[:, None] * K) * v[None, :]
            row_err = float(np.max(np.abs(Q.sum(axis=1) - p0)))
            col_err = float(np.max(np.abs(Q.sum(axis=0) - q0)))
            if max(row_err, col_err) < float(tol):
                ok = True
                break

    Q = (u[:, None] * K) * v[None, :]
    Q = np.maximum(Q, 0.0)
    base = p0[:, None] * q0[None, :]
    mask = Q > 0.0
    kl = float(np.sum(Q[mask] * (np.log(np.maximum(Q[mask], 1e-300)) - np.log(np.maximum(base[mask], 1e-300)))))
    kl = max(kl, 0.0)

    # Envelope gradients.  With Q = diag(u) A diag(v), a valid row-gradient is
    # log(u) - log(p), up to an additive constant.  Same for columns.  The
    # gauge constants are removed by marginal centering.
    gr = np.log(np.maximum(u, 1e-300)) - np.log(np.maximum(p0, 1e-300))
    gc = np.log(np.maximum(v, 1e-300)) - np.log(np.maximum(q0, 1e-300))
    gr -= float(np.sum(p0 * gr))
    gc -= float(np.sum(q0 * gc))
    gr_full[active_r] = gr
    gc_full[active_c] = gc
    return kl, prob_allowed, bool(ok), int(it_done), gr_full, gc_full


def _boundary_bethe_potential_from_rho(
    rho, gamma_by_group, allowed, *, orientation_mode="representative", max_iter=500, tol=1e-10,
):
    """Return self-consistent Bethe potential pi_g and raw KL penalty.

    The raw penalty is the sum over the three positive inter-cube directions.
    The returned pi_g is d(raw_penalty)/d rho_g, centered by rho so that it has
    no irrelevant additive constant in the cube softmax.
    """
    rho = np.maximum(np.asarray(rho, dtype=np.float64), 0.0)
    s = float(rho.sum())
    if s <= 0.0 or not np.isfinite(s):
        return np.zeros_like(rho), np.nan, np.nan, False, 0
    rho = rho / s
    gamma = np.asarray(gamma_by_group, dtype=np.int64)
    n_ft = int(allowed.shape[0])
    pairs = _positive_opposite_face_pairs()
    orient = str(orientation_mode).strip().lower().replace("-", "_")

    pi_g = np.zeros_like(rho, dtype=np.float64)
    raw_total = 0.0
    probs = []
    ok_all = True
    it_total = 0

    if orient in ("representative", "canonical", "directional"):
        for d, od in pairs:
            p_row = np.bincount(gamma[:, d], weights=rho, minlength=n_ft).astype(np.float64)
            p_col = np.bincount(gamma[:, od], weights=rho, minlength=n_ft).astype(np.float64)
            val, prob, good, n_it, gr, gc = _bethe_ipf_penalty_grad_from_marginals(
                p_row, p_col, allowed, max_iter=max_iter, tol=tol
            )
            raw_total += float(val)
            probs.append(float(prob))
            ok_all = bool(ok_all and good)
            it_total += int(n_it)
            pi_g += gr[gamma[:, d]] + gc[gamma[:, od]]
    elif orient in ("orbit", "orbit_average", "rotational", "rotational_average", "rotation_average"):
        # Rotationally averaged face bath.  This removes the arbitrary direction
        # bias of a canonical representative cache, but remains self-consistent:
        # the gradient is distributed back to all six faces of each cube state.
        p_face = np.zeros(n_ft, dtype=np.float64)
        for fd in range(6):
            p_face += np.bincount(gamma[:, fd], weights=rho, minlength=n_ft).astype(np.float64)
        p_face /= 6.0

        face_grad_total = np.zeros(n_ft, dtype=np.float64)
        for _d, _od in pairs:
            val, prob, good, n_it, gr, gc = _bethe_ipf_penalty_grad_from_marginals(
                p_face, p_face, allowed, max_iter=max_iter, tol=tol
            )
            raw_total += float(val)
            probs.append(float(prob))
            ok_all = bool(ok_all and good)
            it_total += int(n_it)
            face_grad_total += gr + gc
        for fd in range(6):
            pi_g += (1.0 / 6.0) * face_grad_total[gamma[:, fd]]
    else:
        raise ValueError("Unknown self-consistent Bethe orientation_mode=%r" % orientation_mode)

    # Remove the additive constant; only relative fields affect rho.
    pi_g -= float(np.sum(rho * pi_g))
    if not probs:
        return pi_g, np.nan, np.nan, False, 0
    return pi_g, max(float(raw_total), 0.0), float(np.mean(probs)), bool(ok_all), int(it_total)


def _infer_gamma_by_group(cache, *, n_groups, n_ft, slots):
    """Infer direction-resolved boundary state Gamma=(N,E,S,W,T,B) per group.

    Preferred source is group_keys with shape (n_groups, 6) in face-type ids or
    shape (n_groups, 24) in raw four-slot faces.  Fallback is patch_group_ptr plus
    patch_to_small when each group has exactly six ordered entries.
    """
    slots = np.asarray(slots, dtype=np.int64)
    slot_to_id = {tuple(int(x) for x in row): int(i) for i, row in enumerate(slots)}

    if "group_keys" in cache:
        keys = np.asarray(cache["group_keys"])
        if keys.ndim == 2 and keys.shape[0] == int(n_groups):
            if keys.shape[1] == 6:
                cand = np.asarray(keys, dtype=np.int64)
                if cand.size == 0 or (np.nanmin(cand) >= 0 and np.nanmax(cand) < int(n_ft)):
                    return np.ascontiguousarray(cand), "group_keys_6_face_ids"
            if keys.shape[1] == 24:
                cand = np.empty((int(n_groups), 6), dtype=np.int64)
                missing = 0
                for d in range(6):
                    block = np.asarray(keys[:, 4*d:4*d+4], dtype=np.int64)
                    for i, row in enumerate(block):
                        val = slot_to_id.get(tuple(int(x) for x in row), -1)
                        if val < 0:
                            missing += 1
                        cand[i, d] = val
                if missing == 0:
                    return np.ascontiguousarray(cand), "group_keys_24_slot_faces"
            if keys.shape[1] > 6:
                cand = np.asarray(keys[:, :6], dtype=np.int64)
                if cand.size == 0 or (np.nanmin(cand) >= 0 and np.nanmax(cand) < int(n_ft)):
                    return np.ascontiguousarray(cand), "group_keys_first6_face_ids"

    if all(k in cache for k in ("patch_group_ptr", "patch_to_small")):
        ptr = np.asarray(cache["patch_group_ptr"], dtype=np.int64)
        pss = np.asarray(cache["patch_to_small"], dtype=np.int64)
        if ptr.ndim == 1 and len(ptr) == int(n_groups) + 1:
            counts = ptr[1:] - ptr[:-1]
            if np.all(counts == 6):
                gamma = np.empty((int(n_groups), 6), dtype=np.int64)
                for g in range(int(n_groups)):
                    gamma[g, :] = pss[int(ptr[g]):int(ptr[g+1])]
                if gamma.size == 0 or (np.nanmin(gamma) >= 0 and np.nanmax(gamma) < int(n_ft)):
                    return np.ascontiguousarray(gamma), "patch_group_ptr_ordered6_fallback"

    return None, "unavailable"


def _boundary_bethe_from_rho(rho, gamma_by_group, allowed, *, correction="boundary_bethe", orientation_mode="orbit_average", max_iter=500, tol=1e-10):
    """Directional boundary-state Bethe penalty from rho_g and Gamma_g.

    The node variable is the full six-face boundary state Gamma.  The hard edge
    support for direction d depends only on Gamma_d and Gamma'_{opp(d)}, so the
    exact I-projection can be evaluated on the compressed face-type marginals
    instead of an impossible dense Gamma x Gamma matrix.
    """
    rho = np.maximum(np.asarray(rho, dtype=np.float64), 0.0)
    s = float(rho.sum())
    if s <= 0.0 or not np.isfinite(s):
        return np.nan, np.nan, False, 0
    rho = rho / s
    gamma = np.asarray(gamma_by_group, dtype=np.int64)
    n_ft = int(allowed.shape[0])
    pairs = _positive_opposite_face_pairs()

    mode = str(correction).strip().lower().replace("-", "_")
    orient = str(orientation_mode).strip().lower().replace("-", "_")

    # Canonical representative caches often store one arbitrary orientation per
    # rotational class.  For a rotationally invariant fluid, the default is to
    # average over the full orientation orbit without explicitly storing 24 rho
    # copies.  Since the edge support depends only on the contacting face, this
    # is exactly the face marginal induced by the orbit-averaged Gamma variable.
    if orient in ("orbit", "orbit_average", "rotational", "rotational_average", "rotation_average"):
        p_face_orbit = np.zeros(n_ft, dtype=np.float64)
        for fd in range(6):
            p_face_orbit += np.bincount(gamma[:, fd], weights=rho, minlength=n_ft).astype(np.float64)
        p_face_orbit /= 6.0
    elif orient in ("representative", "canonical", "directional"):
        p_face_orbit = None
    else:
        raise ValueError("Unknown bethe_orientation_mode=%r; use orbit_average or representative." % orientation_mode)

    raw_total = 0.0
    probs = []
    ok_all = True
    it_total = 0
    for d, od in pairs:
        if p_face_orbit is None:
            p_row = np.bincount(gamma[:, d], weights=rho, minlength=n_ft).astype(np.float64)
            p_col = np.bincount(gamma[:, od], weights=rho, minlength=n_ft).astype(np.float64)
        else:
            p_row = p_face_orbit
            p_col = p_face_orbit
        if mode in ("boundary_entropy", "gamma_entropy", "allowed_probability"):
            val, p_allowed, good = _face_entropy_penalty_from_marginals(p_row, p_col, allowed)
            n_it = 0
        else:
            val, p_allowed, good, n_it = _bethe_ipf_penalty_from_marginals(
                p_row, p_col, allowed, max_iter=max_iter, tol=tol
            )
        raw_total += float(val)
        probs.append(float(p_allowed))
        ok_all = bool(ok_all and good)
        it_total += int(n_it)

    if not probs:
        return np.nan, np.nan, False, 0
    return max(raw_total, 0.0), float(np.mean(probs)), bool(ok_all), int(it_total)


def compute_bethe_curve_correction(
    curve, A_row, e_linear, log_mult, pts, pss, m_patch, delta_np, eps_small,
    *, cache=None, ws=None, correction="none", strength=0.0, contact_factor=1.0,
    threshold=1e-12, compat_mode="slot_exact", orientation_mode="orbit_average", max_iter=500, tol=1e-10,
):
    """Return additive free-energy correction for each phi.

    Modes:
      - boundary_bethe: full Gamma node variable with directional edge supports;
        implemented through the exact compressed face-marginal IPF implied by
        support K(Gamma,Gamma') = allowed[Gamma_d,Gamma'_opp(d)].
      - boundary_entropy: cheaper -log allowed-probability version of the same
        directional Gamma correction.
      - face_bethe / face_entropy: legacy diagnostic using the SAFT association
        face bath pr_s rather than the boundary-state distribution.

    The correction is still perturbative/post-solve: it adds to f(phi) after the
    SAFT-P curve has been solved.  correction='none' or strength=0 exactly
    recovers the uncorrected scanner.
    """
    phi = np.asarray(curve["phi"], dtype=np.float64)
    mus = np.asarray(curve["mus"], dtype=np.float64)
    Ws = np.asarray(curve.get("Ws", np.empty((len(phi), 0))), dtype=np.float64)
    successes = np.asarray(curve["successes"], dtype=bool)
    corr = np.zeros(len(phi), dtype=np.float64)
    raw = np.zeros(len(phi), dtype=np.float64)
    prob = np.ones(len(phi), dtype=np.float64)
    ok = np.zeros(len(phi), dtype=bool)
    iters = np.zeros(len(phi), dtype=np.int32)

    mode = str(correction).strip().lower().replace("-", "_")
    lam = float(strength)
    if mode in ("none", "off", "0") or lam == 0.0:
        ok[:] = True
        return corr, raw, prob, ok, iters, "off"

    slots = np.asarray(ws.slots if ws is not None else cache["small_face_slots"], dtype=np.int64)
    perms4 = np.asarray(ws.perms4 if ws is not None else np.asarray([
        [1, 0, 3, 2], [2, 1, 0, 3], [3, 2, 1, 0], [0, 3, 2, 1]
    ], dtype=np.int64), dtype=np.int64)
    allowed = _build_face_allowed_graph_from_slots(
        slots, perms4, compat_mode=compat_mode, eps_small=eps_small, threshold=threshold
    )

    gamma_by_group = None
    gamma_status = "not_used"
    if mode in ("boundary_bethe", "boundary_ipf", "gamma_bethe", "bethe", "ipf", "boundary_entropy", "gamma_entropy"):
        if cache is None:
            raise ValueError("boundary Bethe correction requires cache to infer Gamma=(N,E,S,W,T,B).")
        gamma_by_group, gamma_status = _infer_gamma_by_group(
            cache, n_groups=len(A_row), n_ft=allowed.shape[0], slots=slots
        )
        if gamma_by_group is None:
            raise ValueError(
                "boundary Bethe correction requested, but the cache does not expose direction-resolved "
                "Gamma faces. Expected group_keys shape (n_groups,6), group_keys shape (n_groups,24), "
                "or patch_group_ptr with exactly six ordered face entries per group. Rebuild the "
                "representative cache with direction-resolved boundary keys."
            )

    for k in range(len(phi)):
        if (not successes[k]) or (not np.isfinite(mus[k])) or Ws.ndim != 2 or Ws.shape[1] == 0:
            corr[k] = np.nan
            raw[k] = np.nan
            prob[k] = np.nan
            ok[k] = False
            continue
        rho, pr = _rho_pr_from_solution_cube_sparse(
            float(mus[k]), np.ascontiguousarray(Ws[k], dtype=np.float64),
            pts, pss, m_patch, delta_np, A_row, e_linear, log_mult,
        )

        if mode in ("boundary_bethe", "boundary_ipf", "gamma_bethe", "bethe", "ipf"):
            val, p_allowed, good, n_it = _boundary_bethe_from_rho(
                rho, gamma_by_group, allowed, correction="boundary_bethe", orientation_mode=orientation_mode, max_iter=max_iter, tol=tol
            )
        elif mode in ("boundary_entropy", "gamma_entropy"):
            val, p_allowed, good, n_it = _boundary_bethe_from_rho(
                rho, gamma_by_group, allowed, correction="boundary_entropy", orientation_mode=orientation_mode, max_iter=max_iter, tol=tol
            )
        elif mode in ("face_entropy", "entropy", "allowed_probability"):
            p = np.asarray(pr, dtype=np.float64)
            psum = float(np.sum(p))
            if psum <= 0.0 or not np.isfinite(psum):
                val, p_allowed, good, n_it = np.nan, np.nan, False, 0
            else:
                val, p_allowed, good = _face_entropy_penalty_from_p(p / psum, allowed)
                n_it = 0
        elif mode in ("face_bethe", "face_ipf", "legacy_face_bethe"):
            p = np.asarray(pr, dtype=np.float64)
            psum = float(np.sum(p))
            if psum <= 0.0 or not np.isfinite(psum):
                val, p_allowed, good, n_it = np.nan, np.nan, False, 0
            else:
                val, p_allowed, good, n_it = _bethe_ipf_penalty_from_p(p / psum, allowed, max_iter=max_iter, tol=tol)
        else:
            raise ValueError(
                f"Unknown bethe_correction={correction!r}; use none, boundary_bethe, "
                "boundary_entropy, face_bethe, or face_entropy."
            )

        raw[k] = val
        prob[k] = p_allowed
        ok[k] = bool(good)
        iters[k] = int(n_it)
        # f is per microscopic lattice site.  A cube has 8 sites; boundary_bethe
        # already sums over the three positive cube-neighbor directions.
        corr[k] = lam * float(contact_factor) * val / 8.0
    return corr, raw, prob, ok, iters, gamma_status

def solve_free_energy_curve_reduced_cube_sparse(
    phi_grid, A_row, e_linear, log_mult,
    pts, pss, m_patch, delta_small_np,
    *, mu_init=0.0, W_init=None, tol=1e-8,
    accept_residual=1e-4, max_iter=30,
    max_jac_rebuilds=1, fallback_newton=False,
):
    """Solve a free-energy curve using Broyden continuation across phi.

    Thermodynamic convention is the corrected Option-B logZ convention:
      e_linear[g] = -logZ_internal_quotiented[g], not <E>_g.

    Broyden strategy:
      - Build a finite-difference inverse Jacobian only when needed.
      - Reuse and rank-1 update that inverse Jacobian across nearby phi values.
      - Rebuild the Jacobian a limited number of times after line-search stalls.
      - Optionally fall back to full Newton for rejected phi values.
    """
    phi_grid = np.asarray(phi_grid, dtype=np.float64)
    A_row = np.ascontiguousarray(A_row, dtype=np.float64)
    e_linear = np.ascontiguousarray(e_linear, dtype=np.float64)
    log_mult = np.ascontiguousarray(log_mult, dtype=np.float64)
    pts = np.asarray(pts)
    pss = np.asarray(pss)
    m_patch = np.asarray(m_patch)
    if not pts.flags.c_contiguous:
        pts = np.ascontiguousarray(pts)
    if not pss.flags.c_contiguous:
        pss = np.ascontiguousarray(pss)
    if not m_patch.flags.c_contiguous:
        m_patch = np.ascontiguousarray(m_patch)
    delta_small_np = np.ascontiguousarray(delta_small_np, dtype=np.float64)

    Fd = int(np.max(pss)) + 1
    phi_lo, phi_hi = float(np.nanmin(A_row)), float(np.nanmax(A_row))
    invalid = (phi_grid < phi_lo - 1e-12) | (phi_grid > phi_hi + 1e-12)
    if np.any(invalid):
        bad = phi_grid[invalid]
        raise ValueError(
            f"solve_free_energy_curve_reduced_cube_sparse received infeasible phi values: "
            f"count={bad.size}, range=[{float(np.nanmin(bad)):.12g},{float(np.nanmax(bad)):.12g}], "
            f"feasible=[{phi_lo:.12g},{phi_hi:.12g}]. Filter phi_grid before calling."
        )

    fvals = np.full(len(phi_grid), np.nan, dtype=np.float64)
    mus = np.full(len(phi_grid), np.nan, dtype=np.float64)
    Ws_store = np.full((len(phi_grid), Fd), np.nan, dtype=np.float64)
    residuals = np.full(len(phi_grid), np.nan, dtype=np.float64)
    successes = np.zeros(len(phi_grid), dtype=bool)

    mu = float(mu_init)
    W = np.zeros(Fd, dtype=np.float64) if W_init is None else np.asarray(W_init, dtype=np.float64).copy()
    B = None

    for k, phi in enumerate(phi_grid):
        phi = float(phi)
        out = _solve_single_phi_broyden(
            phi, mu, W, pts, pss, m_patch, delta_small_np, A_row, e_linear, log_mult,
            tol=tol, max_iter=max_iter, prev_B=B, max_jac_rebuilds=max_jac_rebuilds,
        )

        if (not np.isfinite(out["f"]) or out["rn"] > accept_residual) and fallback_newton:
            out_newton = _solve_single_phi_newton(
                phi, mu, W, pts, pss, m_patch, delta_small_np, A_row, e_linear, log_mult,
                tol=tol, max_iter=max_iter,
            )
            if np.isfinite(out_newton["f"]) and out_newton["rn"] < out["rn"]:
                out = {**out_newton, "B": None}

        residuals[k] = float(out["rn"])
        if np.isfinite(out["f"]) and out["rn"] <= accept_residual:
            mu = float(out["x"][0])
            W = out["x"][1:].copy()
            mus[k] = mu
            Ws_store[k, :] = W
            fvals[k] = out["f"]
            successes[k] = True
            B = out.get("B", None)
        else:
            # Reject this phi but keep sweeping from the last accepted state.
            # Drop B because the inverse Jacobian may now represent a bad branch.
            mus[k] = np.nan
            B = None

    return {
        "phi": phi_grid,
        "f": fvals,
        "mus": mus,
        "Ws": Ws_store,
        "residuals": residuals,
        "successes": successes,
        "mu_W": (float(mu), W.copy()),
    }


def _prepare_boundary_bethe_objects(cache, ws, eps_small, *, compat_mode="attractive", threshold=1e-12):
    slots = np.asarray(ws.slots if ws is not None else cache["small_face_slots"], dtype=np.int64)
    perms4 = np.asarray(ws.perms4 if ws is not None else np.asarray([
        [1, 0, 3, 2], [2, 1, 0, 3], [3, 2, 1, 0], [0, 3, 2, 1]
    ], dtype=np.int64), dtype=np.int64)
    allowed = _build_face_allowed_graph_from_slots(
        slots, perms4, compat_mode=compat_mode, eps_small=eps_small, threshold=threshold
    )
    gamma_by_group, gamma_status = _infer_gamma_by_group(
        cache, n_groups=int(ws.n_groups if ws is not None else cache["n_groups"]),
        n_ft=allowed.shape[0], slots=slots,
    )
    if gamma_by_group is None:
        raise ValueError(
            "selfconsistent_boundary_bethe requires direction-resolved Gamma=(N,E,S,W,T,B). "
            "Expected group_keys shape (n_groups,6), group_keys shape (n_groups,24), "
            "or patch_group_ptr with six ordered boundary entries per group."
        )
    return np.ascontiguousarray(gamma_by_group, dtype=np.int64), allowed, gamma_status




def _build_directed_face_assoc_arrays(cache, ws, eps_face, delta_face):
    """Build a direction-resolved face association closure.

    This is the non-parametric "state/direction dependent" association variant.
    The old closure has one global X_s for each undirected face type s and uses
    the Boltzmann-averaged m_patch counts.  Here each cube class g contributes
    six explicit boundary association sites (N,E,S,W,T,B).  The nonbonded
    fraction is therefore X_{d,f}: it depends on the face direction d and the
    actual face state f carried by the cube.  Equivalently, each cube class g
    samples the six fields X_{d, Gamma_g[d]}.

    No fitted parameter is introduced.  The only association strength is still
    Delta = (exp(-eps_face)-1)/factor.  The directed matrix only allows a face
    in direction d to bind the opposite face direction od.

    Returns:
      pts_dir, pss_dir, m_dir, delta_dir, gamma_status
    where pss_dir = d*n_face + face_id.
    """
    gamma_by_group, gamma_status = _infer_gamma_by_group(
        cache,
        n_groups=int(ws.n_groups if ws is not None else cache["n_groups"]),
        n_ft=int(eps_face.shape[0]),
        slots=np.asarray(ws.slots if ws is not None else cache["small_face_slots"], dtype=np.int64),
    )
    if gamma_by_group is None:
        raise ValueError(
            "directed_face_state association requires direction-resolved Gamma=(N,E,S,W,T,B). "
            "Expected group_keys shape (n_groups,6), group_keys shape (n_groups,24), "
            "or patch_group_ptr with six ordered boundary entries per group."
        )
    gamma_by_group = np.ascontiguousarray(gamma_by_group, dtype=np.int64)
    P = int(gamma_by_group.shape[0])
    n_ft = int(eps_face.shape[0])
    pts_dir = np.repeat(np.arange(P, dtype=np.int64), 6)
    pss_dir = np.empty(6 * P, dtype=np.int64)
    for d in range(6):
        pss_dir[d::6] = d * n_ft + gamma_by_group[:, d]
    m_dir = np.ones(6 * P, dtype=np.float64)

    delta_face = np.asarray(delta_face, dtype=np.float64)
    delta_dir = np.zeros((6 * n_ft, 6 * n_ft), dtype=np.float64)
    try:
        opp = OPPOSITE_FACE
        opp = [int(opp[d]) for d in range(6)]
    except Exception:
        opp = [2, 3, 0, 1, 5, 4]
    for d in range(6):
        od = int(opp[d])
        delta_dir[d*n_ft:(d+1)*n_ft, od*n_ft:(od+1)*n_ft] = delta_face
    return (
        np.ascontiguousarray(pts_dir),
        np.ascontiguousarray(pss_dir),
        np.ascontiguousarray(m_dir),
        np.ascontiguousarray(delta_dir),
        f"directed_face_state:n_face={n_ft};" + str(gamma_status),
    )




def _cube_rotation_face_maps():
    """Return the 24 proper cube rotations as face-direction maps.

    Face convention follows the scanner/builder fallback convention:
        0=N(+y), 1=E(+x), 2=S(-y), 3=W(-x), 4=T(+z), 5=B(-z)

    Each returned map m has m[old_face] = new_face after applying the
    rotation to the cube in the fixed lab frame.
    """
    dirs = np.asarray([
        [0, 1, 0],   # N
        [1, 0, 0],   # E
        [0, -1, 0],  # S
        [-1, 0, 0],  # W
        [0, 0, 1],   # T
        [0, 0, -1],  # B
    ], dtype=np.int64)
    vec_to_face = {tuple(int(x) for x in dirs[i]): i for i in range(6)}

    maps = []
    import itertools as _it
    for perm in _it.permutations(range(3)):
        Pm = np.zeros((3, 3), dtype=np.int64)
        for i, j in enumerate(perm):
            Pm[i, j] = 1
        for signs in _it.product((-1, 1), repeat=3):
            M = Pm.copy()
            for i, sgn in enumerate(signs):
                M[i, :] *= int(sgn)
            # proper rotations only: det = +1. There are 24.
            det = round(float(np.linalg.det(M)))
            if det != 1:
                continue
            mp = np.empty(6, dtype=np.int64)
            for old in range(6):
                new_vec = M @ dirs[old]
                mp[old] = vec_to_face[tuple(int(x) for x in new_vec)]
            maps.append(mp)

    # Deduplicate deterministically.
    uniq = []
    seen = set()
    for m in maps:
        t = tuple(int(x) for x in m)
        if t not in seen:
            seen.add(t)
            uniq.append(m)
    if len(uniq) != 24:
        raise RuntimeError(f"Expected 24 proper cube rotations, got {len(uniq)}")
    return np.asarray(uniq, dtype=np.int64)


def _build_oriented_directed_face_assoc_arrays(cache, ws, eps_face, delta_face):
    """Build the fully oriented-state version of directed-face association.

    This implements Option 1 discussed in the chat: instead of pinning a
    canonical cube class to one arbitrary lab-frame orientation, each canonical
    class g is expanded into 24 oriented states (g,R).  The six face IDs are
    permuted by every proper cube rotation R in the fixed lab frame.

    The solver then treats rho_{g,R} as the population variables and samples
    X_{d, Gamma_{g,R}[d]}.  A compensating log multiplicity -log(24) is returned
    for each oriented copy so that the expansion does not add a spurious pure
    orientation entropy relative to the representative-class scan.  This means
    the expansion only changes the association closure, not the ideal mixing
    reference.

    Current face-state level limitation:
      Face IDs are moved between N/E/S/W/T/B directions.  The in-plane 2x2 slot
      pattern is not additionally rotated because the representative cache does
      not expose the full oriented boundary slots for each rotated class.  This
      is still a strict improvement over pinned representatives, but a builder
      that stores all oriented Gamma states would be cleaner.

    Returns:
      pts_dir, pss_dir, m_dir, delta_dir, log_mult_oriented, n_orient, status
    """
    gamma_by_group, gamma_status = _infer_gamma_by_group(
        cache,
        n_groups=int(ws.n_groups if ws is not None else cache["n_groups"]),
        n_ft=int(eps_face.shape[0]),
        slots=np.asarray(ws.slots if ws is not None else cache["small_face_slots"], dtype=np.int64),
    )
    if gamma_by_group is None:
        raise ValueError(
            "oriented_directed_face_state association requires Gamma=(N,E,S,W,T,B). "
            "Expected group_keys shape (n_groups,6), group_keys shape (n_groups,24), "
            "or patch_group_ptr with six ordered boundary entries per group."
        )

    gamma_by_group = np.ascontiguousarray(gamma_by_group, dtype=np.int64)
    P = int(gamma_by_group.shape[0])
    n_ft = int(eps_face.shape[0])
    face_maps = _cube_rotation_face_maps()
    n_orient = int(face_maps.shape[0])

    gamma_or = np.empty((P * n_orient, 6), dtype=np.int64)
    for g in range(P):
        base = gamma_by_group[g]
        for r in range(n_orient):
            row = g * n_orient + r
            gamma_or[row, :] = -1
            mp = face_maps[r]
            for old_d in range(6):
                new_d = int(mp[old_d])
                gamma_or[row, new_d] = int(base[old_d])
            # Every proper rotation maps all six faces exactly once.
            if np.any(gamma_or[row, :] < 0):
                raise RuntimeError("Internal rotation mapping failure while building oriented Gamma states.")

    P_or = int(gamma_or.shape[0])
    pts_dir = np.repeat(np.arange(P_or, dtype=np.int64), 6)
    pss_dir = np.empty(6 * P_or, dtype=np.int64)
    for d in range(6):
        pss_dir[d::6] = d * n_ft + gamma_or[:, d]
    m_dir = np.ones(6 * P_or, dtype=np.float64)

    delta_face = np.asarray(delta_face, dtype=np.float64)
    delta_dir = np.zeros((6 * n_ft, 6 * n_ft), dtype=np.float64)
    try:
        opp = OPPOSITE_FACE
        opp = [int(opp[d]) for d in range(6)]
    except Exception:
        opp = [2, 3, 0, 1, 5, 4]
    for d in range(6):
        od = int(opp[d])
        delta_dir[d*n_ft:(d+1)*n_ft, od*n_ft:(od+1)*n_ft] = delta_face

    log_mult_oriented = np.full(P_or, -np.log(float(n_orient)), dtype=np.float64)
    status = (
        f"oriented_directed_face_state:n_face={n_ft};n_orient={n_orient};"
        f"face_level_rotation_only;" + str(gamma_status)
    )
    return (
        np.ascontiguousarray(pts_dir),
        np.ascontiguousarray(pss_dir),
        np.ascontiguousarray(m_dir),
        np.ascontiguousarray(delta_dir),
        np.ascontiguousarray(log_mult_oriented),
        n_orient,
        status,
    )



def _build_implicit_oriented_directed_face_assoc_inputs(cache, ws, eps_face, delta_face):
    """Build low-memory inputs for the same oriented directed-face closure.

    This is mathematically equivalent to explicit oriented_directed_face_state
    with 24 copies (g,R) and log multiplicity -log(24), but it does not
    materialize gamma_or, pts_dir, pss_dir, m_dir, or repeated A/e arrays.

    The 24 orientations are summed inside the residual evaluation:
        Z_g^orient = (1/24) sum_R exp[-G_{gR}].
    This preserves the theory while changing only the scanner memory layout.
    """
    gamma_by_group, gamma_status = _infer_gamma_by_group(
        cache,
        n_groups=int(ws.n_groups if ws is not None else cache["n_groups"]),
        n_ft=int(eps_face.shape[0]),
        slots=np.asarray(ws.slots if ws is not None else cache["small_face_slots"], dtype=np.int64),
    )
    if gamma_by_group is None:
        raise ValueError(
            "oriented_directed_face_state association requires Gamma=(N,E,S,W,T,B). "
            "Expected group_keys shape (n_groups,6), group_keys shape (n_groups,24), "
            "or patch_group_ptr with six ordered boundary entries per group."
        )

    n_ft = int(eps_face.shape[0])
    gamma_by_group = np.asarray(gamma_by_group)
    if np.nanmax(gamma_by_group) <= 255 and np.nanmin(gamma_by_group) >= 0:
        gamma_by_group = np.ascontiguousarray(gamma_by_group, dtype=np.uint8)
    else:
        gamma_by_group = np.ascontiguousarray(gamma_by_group, dtype=np.int64)

    face_maps = _cube_rotation_face_maps()
    if np.nanmax(face_maps) <= 255 and np.nanmin(face_maps) >= 0:
        face_maps = np.ascontiguousarray(face_maps, dtype=np.uint8)
    else:
        face_maps = np.ascontiguousarray(face_maps, dtype=np.int64)
    n_orient = int(face_maps.shape[0])

    delta_face = np.asarray(delta_face, dtype=np.float64)
    delta_dir = np.zeros((6 * n_ft, 6 * n_ft), dtype=np.float64)
    try:
        opp = OPPOSITE_FACE
        opp = [int(opp[d]) for d in range(6)]
    except Exception:
        opp = [2, 3, 0, 1, 5, 4]
    for d in range(6):
        od = int(opp[d])
        delta_dir[d*n_ft:(d+1)*n_ft, od*n_ft:(od+1)*n_ft] = delta_face

    P = int(gamma_by_group.shape[0])
    status = (
        f"oriented_directed_face_state:implicit_exact:n_face={n_ft};n_orient={n_orient};"
        f"P={P};avoids_explicit_P_or={P*n_orient};face_level_rotation_only;" + str(gamma_status)
    )
    return gamma_by_group, face_maps, np.ascontiguousarray(delta_dir), n_orient, status


@_njit(cache=True)
def _implicit_oriented_phi_moments_from_c_mu(c, mu, gamma_by_group, face_maps, A, e, lm, n_ft):
    """Return <A> and <A^2> for the implicit oriented ensemble.

    This is used only to find a reasonable initial mu.  It implements the same
    orientation sum as the explicit (g,R) model with -log(n_orient).
    """
    P = A.shape[0]
    n_orient = face_maps.shape[0]
    logits = np.empty(P, dtype=np.float64)
    mx = -1e300

    log_n_orient = np.log(float(n_orient))
    for g in range(P):
        max_y = -1e300
        for r in range(n_orient):
            G = 0.0
            for old_d in range(6):
                new_d = int(face_maps[r, old_d])
                fid = int(gamma_by_group[g, old_d])
                G += c[new_d * n_ft + fid]
            y = -G
            if y > max_y:
                max_y = y
        sum_exp = 0.0
        for r in range(n_orient):
            G = 0.0
            for old_d in range(6):
                new_d = int(face_maps[r, old_d])
                fid = int(gamma_by_group[g, old_d])
                G += c[new_d * n_ft + fid]
            sum_exp += np.exp((-G) - max_y)
        if sum_exp < 1e-300:
            sum_exp = 1e-300
        log_mean_orient = max_y + np.log(sum_exp) - log_n_orient
        v = lm[g] + 8.0 * mu * A[g] - e[g] + log_mean_orient
        logits[g] = v
        if v > mx:
            mx = v

    Z = 0.0
    for g in range(P):
        Z += np.exp(logits[g] - mx)
    if Z < 1e-300:
        Z = 1e-300
    invZ = 1.0 / Z

    phi_mu = 0.0
    second = 0.0
    for g in range(P):
        rho = np.exp(logits[g] - mx) * invZ
        phi_mu += A[g] * rho
        second += A[g] * A[g] * rho
    return phi_mu, second


@_njit(cache=True)
def _implicit_oriented_stats_from_c(c, mu, gamma_by_group, face_maps, A, e, lm, n_ft):
    """Compute rho-weighted statistics without materializing oriented states.

    Equivalent to the explicit model over states (g,R) with log multiplicity
    lm_g - log(n_orient).  Returns phi, directed face abundance pr, mixing
    entropy including conditional orientation entropy, and linear energy.
    """
    P = A.shape[0]
    n_orient = face_maps.shape[0]
    Fd = 6 * n_ft
    logits = np.empty(P, dtype=np.float64)
    pr = np.zeros(Fd, dtype=np.float64)
    mx = -1e300
    log_n_orient = np.log(float(n_orient))

    # Group-level logits: lm + 8 mu A - e + logmean_R exp(-G_gR)
    for g in range(P):
        max_y = -1e300
        for r in range(n_orient):
            G = 0.0
            for old_d in range(6):
                new_d = int(face_maps[r, old_d])
                fid = int(gamma_by_group[g, old_d])
                G += c[new_d * n_ft + fid]
            y = -G
            if y > max_y:
                max_y = y
        sum_exp = 0.0
        for r in range(n_orient):
            G = 0.0
            for old_d in range(6):
                new_d = int(face_maps[r, old_d])
                fid = int(gamma_by_group[g, old_d])
                G += c[new_d * n_ft + fid]
            sum_exp += np.exp((-G) - max_y)
        if sum_exp < 1e-300:
            sum_exp = 1e-300
        log_mean_orient = max_y + np.log(sum_exp) - log_n_orient
        v = lm[g] + 8.0 * mu * A[g] - e[g] + log_mean_orient
        logits[g] = v
        if v > mx:
            mx = v

    Z = 0.0
    for g in range(P):
        Z += np.exp(logits[g] - mx)
    if Z < 1e-300:
        Z = 1e-300
    logZ = mx + np.log(Z)
    invZ_shift = 1.0 / Z

    phi_rho = 0.0
    A_mix = 0.0
    A_lin = 0.0

    for g in range(P):
        rho = np.exp(logits[g] - mx) * invZ_shift
        phi_rho += A[g] * rho
        A_lin += e[g] * rho

        # Base group mixing term.  The conditional orientation entropy below
        # accounts exactly for the explicit copies with lm_g - log(24).
        A_mix += rho * (np.log(max(rho, 1e-300)) - lm[g])

        max_y = -1e300
        for r in range(n_orient):
            G = 0.0
            for old_d in range(6):
                new_d = int(face_maps[r, old_d])
                fid = int(gamma_by_group[g, old_d])
                G += c[new_d * n_ft + fid]
            y = -G
            if y > max_y:
                max_y = y
        sum_exp = 0.0
        for r in range(n_orient):
            G = 0.0
            for old_d in range(6):
                new_d = int(face_maps[r, old_d])
                fid = int(gamma_by_group[g, old_d])
                G += c[new_d * n_ft + fid]
            sum_exp += np.exp((-G) - max_y)
        if sum_exp < 1e-300:
            sum_exp = 1e-300

        orient_entropy = 0.0
        for r in range(n_orient):
            G = 0.0
            for old_d in range(6):
                new_d = int(face_maps[r, old_d])
                fid = int(gamma_by_group[g, old_d])
                G += c[new_d * n_ft + fid]
            q = np.exp((-G) - max_y) / sum_exp
            if q > 0.0:
                orient_entropy += q * np.log(max(float(n_orient) * q, 1e-300))
            wr = rho * q
            for old_d in range(6):
                new_d = int(face_maps[r, old_d])
                fid = int(gamma_by_group[g, old_d])
                pr[new_d * n_ft + fid] += wr

        A_mix += rho * orient_entropy

    return phi_rho, pr, A_mix, A_lin


@_njit(cache=True)
def _eval_residual_implicit_oriented(mu, W, phi, gamma_by_group, face_maps, delta, A, e, lm, n_ft):
    Fd = W.shape[0]

    u = delta @ W
    X = np.empty(Fd, dtype=np.float64)
    hv = np.empty(Fd, dtype=np.float64)
    for s in range(Fd):
        xs = 1.0 / max(1.0 + u[s], 1e-12)
        xs = max(min(xs, 1.0), 1e-12)
        X[s] = xs
        hv[s] = np.log(max(xs, 1e-300)) - 0.5 * xs + 0.5

    _phi0, pr0, _mix0, _lin0 = _implicit_oriented_stats_from_c(
        hv, mu, gamma_by_group, face_maps, A, e, lm, n_ft
    )

    K = np.empty((Fd, Fd), dtype=np.float64)
    for s in range(Fd):
        xs2 = X[s] * X[s]
        for sp in range(Fd):
            K[s, sp] = xs2 * delta[s, sp] * pr0[sp]

    dAdX = np.empty(Fd, dtype=np.float64)
    for s in range(Fd):
        dAdX[s] = pr0[s] * (1.0 / max(X[s], 1e-12) - 0.5)

    Msys = np.empty((Fd, Fd), dtype=np.float64)
    for s in range(Fd):
        for sp in range(Fd):
            Msys[s, sp] = K[sp, s]
        Msys[s, s] += 1.0 + 1e-12

    alpha = np.linalg.solve(Msys, dAdX)

    beta = np.zeros(Fd, dtype=np.float64)
    for s in range(Fd):
        acc = 0.0
        for sp in range(Fd):
            acc += delta[sp, s] * alpha[sp] * X[sp] * X[sp]
        beta[s] = -acc

    c = np.empty(Fd, dtype=np.float64)
    for s in range(Fd):
        c[s] = hv[s] + beta[s] * X[s]

    phi_rho, pr, A_mix, A_lin = _implicit_oriented_stats_from_c(
        c, mu, gamma_by_group, face_maps, A, e, lm, n_ft
    )

    R = np.empty(1 + Fd, dtype=np.float64)
    R[0] = phi_rho - phi
    for s in range(Fd):
        R[1 + s] = W[s] - X[s] * pr[s]

    A_assoc = 0.0
    for s in range(Fd):
        A_assoc += hv[s] * pr[s]

    f = (A_mix + A_lin + A_assoc) / 8.0
    return R, f


@_njit(cache=True)
def _find_initial_mu_implicit_oriented(phi_target, W, gamma_by_group, face_maps, delta, A, e, lm, n_ft):
    Fd = W.shape[0]
    u = delta @ W
    hv = np.empty(Fd, dtype=np.float64)
    for s in range(Fd):
        xs = 1.0 / max(1.0 + u[s], 1e-12)
        xs = max(min(xs, 1.0), 1e-12)
        hv[s] = np.log(max(xs, 1e-300)) - 0.5 * xs + 0.5

    mu_lo = -100.0
    mu_hi = 100.0
    mu = 0.0
    for _ in range(120):
        phi_mu, second = _implicit_oriented_phi_moments_from_c_mu(
            hv, mu, gamma_by_group, face_maps, A, e, lm, n_ft
        )
        dphi = 8.0 * (second - phi_mu * phi_mu)
        err = phi_mu - phi_target
        if abs(err) < 1e-12:
            break
        if err > 0.0:
            mu_hi = mu
        else:
            mu_lo = mu
        if abs(dphi) > 1e-30:
            mu_new = mu - err / dphi
            if mu_new < mu_lo or mu_new > mu_hi:
                mu_new = 0.5 * (mu_lo + mu_hi)
        else:
            mu_new = 0.5 * (mu_lo + mu_hi)
        mu = mu_new
    return mu


def _build_fd_jacobian_implicit_oriented(mu, W, phi, gamma_by_group, face_maps, delta, A, e, lm, n_ft, R, h_fd=1e-7):
    Fd = W.shape[0]
    n = 1 + Fd
    Jac = np.empty((n, n), dtype=np.float64)

    Rp, _ = _eval_residual_implicit_oriented(
        mu + h_fd, W, phi, gamma_by_group, face_maps, delta, A, e, lm, n_ft)
    Jac[:, 0] = (Rp - R) / h_fd

    for j in range(Fd):
        Wp = W.copy()
        Wp[j] += h_fd
        Rp, _ = _eval_residual_implicit_oriented(
            mu, Wp, phi, gamma_by_group, face_maps, delta, A, e, lm, n_ft)
        Jac[:, 1 + j] = (Rp - R) / h_fd

    Jac[np.diag_indices_from(Jac)] += 1e-12
    return Jac


def _solve_single_phi_broyden_implicit_oriented(
    phi, mu0, W0, gamma_by_group, face_maps, delta, A, e, lm, n_ft,
    *, tol=1e-8, max_iter=30, step_cap=10.0, h_fd=1e-7,
    prev_B=None, max_jac_rebuilds=1,
):
    Fd = W0.shape[0]
    n = 1 + Fd

    mu = float(mu0)
    W = W0.copy()

    R, f = _eval_residual_implicit_oriented(mu, W, phi, gamma_by_group, face_maps, delta, A, e, lm, n_ft)
    rn = float(np.linalg.norm(R))

    if rn > 0.1:
        mu = _find_initial_mu_implicit_oriented(phi, W, gamma_by_group, face_maps, delta, A, e, lm, n_ft)
        R, f = _eval_residual_implicit_oriented(mu, W, phi, gamma_by_group, face_maps, delta, A, e, lm, n_ft)
        rn = float(np.linalg.norm(R))

    best_x = np.empty(n, dtype=np.float64)
    best_x[0] = mu
    best_x[1:] = W.copy()
    best_rn = rn
    best_f = float(f) if np.isfinite(f) else np.nan

    if rn < tol:
        return {"x": best_x, "rn": best_rn, "f": best_f, "success": True, "B": prev_B}

    n_jac_builds = 0
    if prev_B is not None:
        B = prev_B.copy()
    else:
        Jac = _build_fd_jacobian_implicit_oriented(mu, W, phi, gamma_by_group, face_maps, delta, A, e, lm, n_ft, R, h_fd)
        try:
            B = np.linalg.inv(Jac)
        except np.linalg.LinAlgError:
            B = np.eye(n, dtype=np.float64)
        n_jac_builds = 1

    stall_count = 0
    for _it in range(max_iter):
        dx = -(B @ R)
        dx_norm = float(np.linalg.norm(dx))
        if dx_norm > step_cap:
            dx *= step_cap / dx_norm

        alpha = 1.0
        R_new = None
        f_new = np.nan
        rn_new = rn
        for _ in range(4):
            mu_try = mu + alpha * dx[0]
            W_try = W + alpha * dx[1:]
            R_try, f_try = _eval_residual_implicit_oriented(
                mu_try, W_try, phi, gamma_by_group, face_maps, delta, A, e, lm, n_ft)
            rn_try = float(np.linalg.norm(R_try))
            if rn_try < rn:
                R_new = R_try
                f_new = f_try
                rn_new = rn_try
                break
            alpha *= 0.5

        if R_new is None:
            stall_count += 1
            if stall_count >= 3 and n_jac_builds < max_jac_rebuilds:
                Jac = _build_fd_jacobian_implicit_oriented(mu, W, phi, gamma_by_group, face_maps, delta, A, e, lm, n_ft, R, h_fd)
                try:
                    B = np.linalg.inv(Jac)
                except np.linalg.LinAlgError:
                    break
                n_jac_builds += 1
                stall_count = 0
            elif stall_count >= 3:
                break
            continue

        stall_count = 0
        s = alpha * dx
        y = R_new - R
        By = B @ y
        sTBy = float(s @ By)
        if abs(sTBy) > 1e-30:
            B += np.outer(s - By, s @ B) / sTBy

        mu += alpha * dx[0]
        W += alpha * dx[1:]
        R = R_new
        f = f_new
        rn = rn_new

        if rn < best_rn:
            best_x[0] = mu
            best_x[1:] = W.copy()
            best_rn = rn
            best_f = float(f)

        if rn < tol:
            break

    return {"x": best_x, "rn": best_rn, "f": best_f, "success": best_rn <= tol, "B": B}


def _solve_single_phi_newton_implicit_oriented(
    phi, mu0, W0, gamma_by_group, face_maps, delta, A, e, lm, n_ft,
    *, tol=1e-8, max_iter=30, step_cap=10.0, h_fd=1e-7,
):
    Fd = W0.shape[0]
    n = 1 + Fd
    mu = float(mu0)
    W = W0.copy()
    R, f = _eval_residual_implicit_oriented(mu, W, phi, gamma_by_group, face_maps, delta, A, e, lm, n_ft)
    rn = float(np.linalg.norm(R))
    if rn > 0.1:
        mu = _find_initial_mu_implicit_oriented(phi, W, gamma_by_group, face_maps, delta, A, e, lm, n_ft)
        R, f = _eval_residual_implicit_oriented(mu, W, phi, gamma_by_group, face_maps, delta, A, e, lm, n_ft)
        rn = float(np.linalg.norm(R))

    best_x = np.empty(n, dtype=np.float64)
    best_x[0] = mu
    best_x[1:] = W.copy()
    best_rn = rn
    best_f = float(f) if np.isfinite(f) else np.nan

    for _it in range(max_iter):
        if rn < tol:
            break
        Jac = _build_fd_jacobian_implicit_oriented(mu, W, phi, gamma_by_group, face_maps, delta, A, e, lm, n_ft, R, h_fd)
        try:
            dp = np.linalg.solve(Jac, -R)
        except np.linalg.LinAlgError:
            break
        np.clip(dp, -step_cap, step_cap, out=dp)
        alpha = 1.0
        for _ in range(8):
            R_try, f_try = _eval_residual_implicit_oriented(
                mu + alpha * dp[0], W + alpha * dp[1:], phi, gamma_by_group, face_maps, delta, A, e, lm, n_ft)
            if float(np.linalg.norm(R_try)) < rn:
                break
            alpha *= 0.5
        mu += alpha * dp[0]
        W += alpha * dp[1:]
        R, f = _eval_residual_implicit_oriented(mu, W, phi, gamma_by_group, face_maps, delta, A, e, lm, n_ft)
        rn = float(np.linalg.norm(R))
        if rn < best_rn:
            best_x[0] = mu
            best_x[1:] = W.copy()
            best_rn = rn
            best_f = float(f)
    return {"x": best_x, "rn": best_rn, "f": best_f, "success": best_rn <= tol}


def solve_free_energy_curve_reduced_cube_implicit_oriented(
    phi_grid, A_row, e_linear, log_mult,
    gamma_by_group, face_maps, delta_small_np, n_ft,
    *, mu_init=0.0, W_init=None, tol=1e-8,
    accept_residual=1e-4, max_iter=30,
    max_jac_rebuilds=1, fallback_newton=False,
):
    """Same oriented directed-face theory as explicit (g,R), with O(P) memory.

    This solver sums over the 24 cube orientations inside the residual.  It is
    intended to replace explicit oriented_directed_face_state for large caches.
    """
    phi_grid = np.asarray(phi_grid, dtype=np.float64)
    A_row = np.ascontiguousarray(A_row, dtype=np.float64)
    e_linear = np.ascontiguousarray(e_linear, dtype=np.float64)
    log_mult = np.ascontiguousarray(log_mult, dtype=np.float64)
    gamma_by_group = np.ascontiguousarray(gamma_by_group)
    face_maps = np.ascontiguousarray(face_maps)
    delta_small_np = np.ascontiguousarray(delta_small_np, dtype=np.float64)

    Fd = int(delta_small_np.shape[0])
    phi_lo, phi_hi = float(np.nanmin(A_row)), float(np.nanmax(A_row))
    invalid = (phi_grid < phi_lo - 1e-12) | (phi_grid > phi_hi + 1e-12)
    if np.any(invalid):
        bad = phi_grid[invalid]
        raise ValueError(
            f"solve_free_energy_curve_reduced_cube_implicit_oriented received infeasible phi values: "
            f"count={bad.size}, range=[{float(np.nanmin(bad)):.12g},{float(np.nanmax(bad)):.12g}], "
            f"feasible=[{phi_lo:.12g},{phi_hi:.12g}]. Filter phi_grid before calling."
        )

    fvals = np.full(len(phi_grid), np.nan, dtype=np.float64)
    mus = np.full(len(phi_grid), np.nan, dtype=np.float64)
    Ws_store = np.full((len(phi_grid), Fd), np.nan, dtype=np.float64)
    residuals = np.full(len(phi_grid), np.nan, dtype=np.float64)
    successes = np.zeros(len(phi_grid), dtype=bool)

    mu = float(mu_init)
    W = np.zeros(Fd, dtype=np.float64) if W_init is None else np.asarray(W_init, dtype=np.float64).copy()
    B = None

    for k, phi in enumerate(phi_grid):
        phi = float(phi)
        out = _solve_single_phi_broyden_implicit_oriented(
            phi, mu, W, gamma_by_group, face_maps, delta_small_np, A_row, e_linear, log_mult, int(n_ft),
            tol=tol, max_iter=max_iter, prev_B=B, max_jac_rebuilds=max_jac_rebuilds,
        )

        if (not np.isfinite(out["f"]) or out["rn"] > accept_residual) and fallback_newton:
            out_newton = _solve_single_phi_newton_implicit_oriented(
                phi, mu, W, gamma_by_group, face_maps, delta_small_np, A_row, e_linear, log_mult, int(n_ft),
                tol=tol, max_iter=max_iter,
            )
            if np.isfinite(out_newton["f"]) and out_newton["rn"] < out["rn"]:
                out = {**out_newton, "B": None}

        residuals[k] = float(out["rn"])
        if np.isfinite(out["f"]) and out["rn"] <= accept_residual:
            mu = float(out["x"][0])
            W = out["x"][1:].copy()
            mus[k] = mu
            Ws_store[k, :] = W
            fvals[k] = out["f"]
            successes[k] = True
            B = out.get("B", None)
        else:
            mus[k] = np.nan
            B = None

    return {
        "phi": phi_grid,
        "f": fvals,
        "mus": mus,
        "Ws": Ws_store,
        "residuals": residuals,
        "successes": successes,
        "mu_W": (float(mu), W.copy()),
    }


def solve_free_energy_curve_reduced_cube_sparse_selfconsistent_bethe(
    phi_grid, A_row, e_linear, log_mult,
    pts, pss, m_patch, delta_small_np,
    *, cache, ws, eps_small,
    compat_mode="attractive", orientation_mode="orbit_average", threshold=1e-12,
    bethe_max_iter=500, bethe_tol=1e-10,
    sc_max_iter=20, sc_tol=1e-5, sc_damping=0.5,
    mu_init=0.0, W_init=None, tol=1e-8,
    accept_residual=1e-4, max_iter=30,
    fallback_newton=False,
):
    """Self-consistent boundary-state Bethe/cavity SAFT-P curve.

    This is not a post-solve correction.  At each target phi it iterates:
      1. solve the reduced SAFT-P equations with an external cube-state field pi_g;
      2. reconstruct rho_g;
      3. solve the boundary compatibility IPF problem and compute pi_g=dF_B/d rho_g;
      4. repeat until pi_g is self-consistent.

    The final reported free energy is
        f = f_SAFT-P + raw_boundary_Bethe / 8.
    The coefficient is fixed by the cube size.  No empirical lambda appears.
    """
    phi_grid = np.asarray(phi_grid, dtype=np.float64)
    A_row = np.ascontiguousarray(A_row, dtype=np.float64)
    e_linear = np.ascontiguousarray(e_linear, dtype=np.float64)
    log_mult = np.ascontiguousarray(log_mult, dtype=np.float64)
    pts = np.ascontiguousarray(np.asarray(pts))
    pss = np.ascontiguousarray(np.asarray(pss))
    m_patch = np.ascontiguousarray(np.asarray(m_patch))
    delta_small_np = np.ascontiguousarray(delta_small_np, dtype=np.float64)

    Fd = int(np.max(pss)) + 1
    P = int(A_row.shape[0])
    gamma_by_group, allowed, gamma_status = _prepare_boundary_bethe_objects(
        cache, ws, eps_small, compat_mode=compat_mode, threshold=threshold
    )

    phi_lo, phi_hi = float(np.nanmin(A_row)), float(np.nanmax(A_row))
    invalid = (phi_grid < phi_lo - 1e-12) | (phi_grid > phi_hi + 1e-12)
    if np.any(invalid):
        bad = phi_grid[invalid]
        raise ValueError(
            f"selfconsistent solver received infeasible phi values: count={bad.size}, "
            f"range=[{float(np.nanmin(bad)):.12g},{float(np.nanmax(bad)):.12g}], "
            f"feasible=[{phi_lo:.12g},{phi_hi:.12g}]."
        )

    fvals = np.full(len(phi_grid), np.nan, dtype=np.float64)
    f_base_vals = np.full(len(phi_grid), np.nan, dtype=np.float64)
    mus = np.full(len(phi_grid), np.nan, dtype=np.float64)
    Ws_store = np.full((len(phi_grid), Fd), np.nan, dtype=np.float64)
    residuals = np.full(len(phi_grid), np.nan, dtype=np.float64)
    successes = np.zeros(len(phi_grid), dtype=bool)
    raw_store = np.full(len(phi_grid), np.nan, dtype=np.float64)
    corr_store = np.full(len(phi_grid), np.nan, dtype=np.float64)
    prob_store = np.full(len(phi_grid), np.nan, dtype=np.float64)
    ok_store = np.zeros(len(phi_grid), dtype=bool)
    sc_iters_store = np.zeros(len(phi_grid), dtype=np.int32)
    ipf_iters_store = np.zeros(len(phi_grid), dtype=np.int32)
    pi_diff_store = np.full(len(phi_grid), np.nan, dtype=np.float64)

    mu = float(mu_init)
    W = np.zeros(Fd, dtype=np.float64) if W_init is None else np.asarray(W_init, dtype=np.float64).copy()
    pi_prev = np.zeros(P, dtype=np.float64)

    sc_damping = float(sc_damping)
    sc_damping = min(max(sc_damping, 1e-6), 1.0)

    for k, phi in enumerate(phi_grid):
        phi = float(phi)
        pi_ext = pi_prev.copy()
        best_out = None
        last_raw = np.nan
        last_prob = np.nan
        last_ok = False
        last_ipf = 0
        last_diff = np.nan

        for sc_it in range(int(sc_max_iter)):
            out = _solve_single_phi_broyden_field(
                phi, mu, W, pts, pss, m_patch, delta_small_np, A_row, e_linear, log_mult, pi_ext,
                tol=tol, max_iter=max_iter, max_jac_rebuilds=0,
            )
            best_out = out
            if (not np.isfinite(out["f"])) or out["rn"] > accept_residual:
                break
            mu_sc = float(out["x"][0])
            W_sc = out["x"][1:].copy()
            rho, _pr = _rho_pr_from_solution_cube_sparse_field(
                mu_sc, W_sc, pts, pss, m_patch, delta_small_np, A_row, e_linear, log_mult, pi_ext
            )
            pi_new, raw, prob, good, n_ipf = _boundary_bethe_potential_from_rho(
                rho, gamma_by_group, allowed,
                orientation_mode=orientation_mode, max_iter=bethe_max_iter, tol=bethe_tol,
            )
            if not np.all(np.isfinite(pi_new)):
                pi_new = np.nan_to_num(pi_new, nan=0.0, posinf=0.0, neginf=0.0)
            # Stabilize: remove arbitrary constant and damp the cavity-field update.
            pi_new -= float(np.sum(rho * pi_new))
            last_diff = float(np.max(np.abs(pi_new - pi_ext)))
            pi_ext = (1.0 - sc_damping) * pi_ext + sc_damping * pi_new
            pi_ext -= float(np.sum(rho * pi_ext))
            last_raw, last_prob, last_ok, last_ipf = float(raw), float(prob), bool(good), int(n_ipf)
            if last_diff < float(sc_tol):
                break

        # Final solve using the last self-consistent field.
        out = _solve_single_phi_broyden_field(
            phi, mu, W, pts, pss, m_patch, delta_small_np, A_row, e_linear, log_mult, pi_ext,
            tol=tol, max_iter=max_iter, max_jac_rebuilds=0,
        )
        residuals[k] = float(out["rn"])
        if np.isfinite(out["f"]) and out["rn"] <= accept_residual:
            mu = float(out["x"][0])
            W = out["x"][1:].copy()
            rho, _pr = _rho_pr_from_solution_cube_sparse_field(
                mu, W, pts, pss, m_patch, delta_small_np, A_row, e_linear, log_mult, pi_ext
            )
            pi_final, raw, prob, good, n_ipf = _boundary_bethe_potential_from_rho(
                rho, gamma_by_group, allowed,
                orientation_mode=orientation_mode, max_iter=bethe_max_iter, tol=bethe_tol,
            )
            raw = float(raw)
            corr = raw / 8.0
            mus[k] = mu
            Ws_store[k, :] = W
            f_base_vals[k] = float(out["f"])
            fvals[k] = float(out["f"] + corr)
            successes[k] = True
            raw_store[k] = raw
            corr_store[k] = corr
            prob_store[k] = float(prob)
            ok_store[k] = bool(good)
            sc_iters_store[k] = int(sc_it + 1) if 'sc_it' in locals() else 0
            ipf_iters_store[k] = int(n_ipf)
            pi_diff_store[k] = float(last_diff) if np.isfinite(last_diff) else np.nan
            pi_prev = pi_ext.copy()
        else:
            mus[k] = np.nan
            pi_prev[:] = 0.0

    return {
        "phi": phi_grid,
        "f": fvals,
        "f_base": f_base_vals,
        "mus": mus,
        "Ws": Ws_store,
        "residuals": residuals,
        "successes": successes,
        "mu_W": (float(mu), W.copy()),
        "bethe_raw": raw_store,
        "bethe_corr": corr_store,
        "bethe_prob": prob_store,
        "bethe_ok": ok_store,
        "bethe_sc_iters": sc_iters_store,
        "bethe_ipf_iters": ipf_iters_store,
        "bethe_pi_diff": pi_diff_store,
        "bethe_gamma_status": gamma_status,
    }




# ===================================================================
# Exact composition-field treatment inside each cube class
# ===================================================================

@_njit(cache=True)
def _internal_logz_occ_stats(nu, logZ_by_nocc):
    """Evaluate the nu-tilted internal partition function for every class.

    logZ_by_nocc[g,n] stores
        log sum_{c in g, Nocc(c)=n} exp[-E_c]
    including the class-level rotational quotient.  Therefore
        log Z_g(nu) = log sum_n exp[logZ_by_nocc[g,n] + nu*n].

    Returns logZ_g(nu), <Nocc>/8, and <(Nocc/8)^2> within each class.
    """
    P = logZ_by_nocc.shape[0]
    Kocc = logZ_by_nocc.shape[1]
    logZ = np.empty(P, dtype=np.float64)
    A = np.empty(P, dtype=np.float64)
    A2 = np.empty(P, dtype=np.float64)
    for g in range(P):
        mx = -1e300
        for n in range(Kocc):
            b = logZ_by_nocc[g, n]
            if np.isfinite(b):
                v = b + nu * n
                if v > mx:
                    mx = v
        if mx <= -1e299:
            logZ[g] = -1e300
            A[g] = 0.0
            A2[g] = 0.0
            continue
        z = 0.0
        z1 = 0.0
        z2 = 0.0
        for n in range(Kocc):
            b = logZ_by_nocc[g, n]
            if np.isfinite(b):
                w = np.exp(b + nu * n - mx)
                z += w
                z1 += n * w
                z2 += n * n * w
        if z < 1e-300:
            z = 1e-300
        logZ[g] = mx + np.log(z)
        invz = 1.0 / z
        A[g] = (z1 * invz) / 8.0
        A2[g] = (z2 * invz) / 64.0
    return logZ, A, A2


@_njit(cache=True)
def _eval_residual_reduced_cube_sparse_nu(
    nu, W, phi, pts, pss, m_patch, delta, logZ_by_nocc, lm
):
    """Reduced residual with nu inside the internal class partition function.

    The class populations obey
        psi_g proportional to exp[lm_g + log Z_g(nu) - (M c)_g],
    while the composition residual uses the nu-dependent class occupancy
        A_g(nu)=<Nocc>_g/8.
    The returned free energy is the physical Helmholtz free energy, not the
    nu-tilted Legendre functional.
    """
    Fd = W.shape[0]
    P = logZ_by_nocc.shape[0]

    logZg, A, _A2 = _internal_logz_occ_stats(nu, logZ_by_nocc)

    u = delta @ W
    X = np.empty(Fd, dtype=np.float64)
    hv = np.empty(Fd, dtype=np.float64)
    for s in range(Fd):
        xs = 1.0 / max(1.0 + u[s], 1e-12)
        xs = max(min(xs, 1.0), 1e-12)
        X[s] = xs
        hv[s] = np.log(max(xs, 1e-300)) - 0.5 * xs + 0.5

    g0 = _apply_g_from_c(hv, pts, pss, m_patch, P)
    logits = np.empty(P, dtype=np.float64)
    mx = -1e300
    for i in range(P):
        v = lm[i] + logZg[i] - g0[i]
        logits[i] = v
        if v > mx:
            mx = v

    rho = np.empty(P, dtype=np.float64)
    Z = 0.0
    for i in range(P):
        ri = np.exp(logits[i] - mx)
        rho[i] = ri
        Z += ri
    if Z < 1e-300:
        Z = 1e-300
    invZ = 1.0 / Z
    for i in range(P):
        rho[i] *= invZ

    pr = _apply_pr_from_rho(rho, pts, pss, m_patch, Fd)

    K = np.empty((Fd, Fd), dtype=np.float64)
    for s in range(Fd):
        xs2 = X[s] * X[s]
        for sp in range(Fd):
            K[s, sp] = xs2 * delta[s, sp] * pr[sp]

    dAdX = np.empty(Fd, dtype=np.float64)
    for s in range(Fd):
        dAdX[s] = pr[s] * (1.0 / max(X[s], 1e-12) - 0.5)

    Msys = np.empty((Fd, Fd), dtype=np.float64)
    for s in range(Fd):
        for sp in range(Fd):
            Msys[s, sp] = K[sp, s]
        Msys[s, s] += 1.0 + 1e-12

    alpha = np.linalg.solve(Msys, dAdX)

    beta = np.zeros(Fd, dtype=np.float64)
    for s in range(Fd):
        acc = 0.0
        for sp in range(Fd):
            acc += delta[sp, s] * alpha[sp] * X[sp] * X[sp]
        beta[s] = -acc

    c = np.empty(Fd, dtype=np.float64)
    for s in range(Fd):
        c[s] = hv[s] + beta[s] * X[s]

    g = _apply_g_from_c(c, pts, pss, m_patch, P)
    mx = -1e300
    for i in range(P):
        v = lm[i] + logZg[i] - g[i]
        logits[i] = v
        if v > mx:
            mx = v

    Z = 0.0
    for i in range(P):
        ri = np.exp(logits[i] - mx)
        rho[i] = ri
        Z += ri
    if Z < 1e-300:
        Z = 1e-300
    invZ = 1.0 / Z
    for i in range(P):
        rho[i] *= invZ

    pr = _apply_pr_from_rho(rho, pts, pss, m_patch, Fd)

    R = np.empty(1 + Fd, dtype=np.float64)
    phi_rho = 0.0
    for i in range(P):
        phi_rho += A[i] * rho[i]
    R[0] = phi_rho - phi
    for s in range(Fd):
        R[1 + s] = W[s] - X[s] * pr[s]

    A_mix = 0.0
    A_internal = 0.0
    for i in range(P):
        A_mix += rho[i] * (np.log(max(rho[i], 1e-300)) - lm[i])
        # q_{c|g}(nu) minimizes the tilted functional.  The physical internal
        # contribution is g_g(nu)+nu*<Nocc>_g = -logZ_g+8 nu A_g.
        A_internal += rho[i] * (-logZg[i] + 8.0 * nu * A[i])

    A_assoc = 0.0
    for s in range(Fd):
        A_assoc += hv[s] * pr[s]

    f = (A_mix + A_internal + A_assoc) / 8.0
    return R, f


@_njit(cache=True)
def _find_initial_nu_cube_sparse(
    phi_target, W, pts, pss, m_patch, delta, logZ_by_nocc, lm
):
    Fd = W.shape[0]
    P = logZ_by_nocc.shape[0]
    u = delta @ W
    hv = np.empty(Fd, dtype=np.float64)
    for s in range(Fd):
        xs = 1.0 / max(1.0 + u[s], 1e-12)
        xs = max(min(xs, 1.0), 1e-12)
        hv[s] = np.log(max(xs, 1e-300)) - 0.5 * xs + 0.5
    g = _apply_g_from_c(hv, pts, pss, m_patch, P)

    lo = -100.0
    hi = 100.0
    nu = 0.0
    for _ in range(140):
        logZg, A, _A2 = _internal_logz_occ_stats(nu, logZ_by_nocc)
        mx = -1e300
        for i in range(P):
            v = lm[i] + logZg[i] - g[i]
            if v > mx:
                mx = v
        Z = 0.0
        phi_nu = 0.0
        for i in range(P):
            w = np.exp(lm[i] + logZg[i] - g[i] - mx)
            Z += w
            phi_nu += A[i] * w
        if Z < 1e-300:
            Z = 1e-300
        phi_nu /= Z
        if abs(phi_nu - phi_target) < 1e-12:
            break
        if phi_nu > phi_target:
            hi = nu
        else:
            lo = nu
        nu = 0.5 * (lo + hi)
    return nu


@_njit(cache=True)
def _implicit_oriented_phi_from_c_nu(
    c, nu, gamma_by_group, face_maps, logZ_by_nocc, lm, n_ft
):
    P = logZ_by_nocc.shape[0]
    n_orient = face_maps.shape[0]
    logZg, A, _A2 = _internal_logz_occ_stats(nu, logZ_by_nocc)
    log_n_orient = np.log(float(n_orient))
    logits = np.empty(P, dtype=np.float64)
    mx = -1e300
    for g in range(P):
        max_y = -1e300
        for r in range(n_orient):
            G = 0.0
            for old_d in range(6):
                new_d = int(face_maps[r, old_d])
                fid = int(gamma_by_group[g, old_d])
                G += c[new_d * n_ft + fid]
            y = -G
            if y > max_y:
                max_y = y
        sum_exp = 0.0
        for r in range(n_orient):
            G = 0.0
            for old_d in range(6):
                new_d = int(face_maps[r, old_d])
                fid = int(gamma_by_group[g, old_d])
                G += c[new_d * n_ft + fid]
            sum_exp += np.exp((-G) - max_y)
        if sum_exp < 1e-300:
            sum_exp = 1e-300
        log_mean_orient = max_y + np.log(sum_exp) - log_n_orient
        v = lm[g] + logZg[g] + log_mean_orient
        logits[g] = v
        if v > mx:
            mx = v
    Z = 0.0
    phi_nu = 0.0
    for g in range(P):
        w = np.exp(logits[g] - mx)
        Z += w
        phi_nu += A[g] * w
    if Z < 1e-300:
        Z = 1e-300
    return phi_nu / Z


@_njit(cache=True)
def _implicit_oriented_stats_from_c_nu(
    c, nu, gamma_by_group, face_maps, logZ_by_nocc, lm, n_ft
):
    P = logZ_by_nocc.shape[0]
    n_orient = face_maps.shape[0]
    Fd = 6 * n_ft
    logZg, A, _A2 = _internal_logz_occ_stats(nu, logZ_by_nocc)
    logits = np.empty(P, dtype=np.float64)
    pr = np.zeros(Fd, dtype=np.float64)
    mx = -1e300
    log_n_orient = np.log(float(n_orient))

    for g in range(P):
        max_y = -1e300
        for r in range(n_orient):
            G = 0.0
            for old_d in range(6):
                new_d = int(face_maps[r, old_d])
                fid = int(gamma_by_group[g, old_d])
                G += c[new_d * n_ft + fid]
            y = -G
            if y > max_y:
                max_y = y
        sum_exp = 0.0
        for r in range(n_orient):
            G = 0.0
            for old_d in range(6):
                new_d = int(face_maps[r, old_d])
                fid = int(gamma_by_group[g, old_d])
                G += c[new_d * n_ft + fid]
            sum_exp += np.exp((-G) - max_y)
        if sum_exp < 1e-300:
            sum_exp = 1e-300
        log_mean_orient = max_y + np.log(sum_exp) - log_n_orient
        v = lm[g] + logZg[g] + log_mean_orient
        logits[g] = v
        if v > mx:
            mx = v

    Z = 0.0
    for g in range(P):
        Z += np.exp(logits[g] - mx)
    if Z < 1e-300:
        Z = 1e-300
    invZ = 1.0 / Z

    phi_rho = 0.0
    A_mix = 0.0
    A_internal = 0.0

    for g in range(P):
        rho = np.exp(logits[g] - mx) * invZ
        phi_rho += A[g] * rho
        A_internal += rho * (-logZg[g] + 8.0 * nu * A[g])
        A_mix += rho * (np.log(max(rho, 1e-300)) - lm[g])

        max_y = -1e300
        for r in range(n_orient):
            G = 0.0
            for old_d in range(6):
                new_d = int(face_maps[r, old_d])
                fid = int(gamma_by_group[g, old_d])
                G += c[new_d * n_ft + fid]
            y = -G
            if y > max_y:
                max_y = y
        sum_exp = 0.0
        for r in range(n_orient):
            G = 0.0
            for old_d in range(6):
                new_d = int(face_maps[r, old_d])
                fid = int(gamma_by_group[g, old_d])
                G += c[new_d * n_ft + fid]
            sum_exp += np.exp((-G) - max_y)
        if sum_exp < 1e-300:
            sum_exp = 1e-300

        orient_entropy = 0.0
        for r in range(n_orient):
            G = 0.0
            for old_d in range(6):
                new_d = int(face_maps[r, old_d])
                fid = int(gamma_by_group[g, old_d])
                G += c[new_d * n_ft + fid]
            q = np.exp((-G) - max_y) / sum_exp
            if q > 0.0:
                orient_entropy += q * np.log(max(float(n_orient) * q, 1e-300))
            wr = rho * q
            for old_d in range(6):
                new_d = int(face_maps[r, old_d])
                fid = int(gamma_by_group[g, old_d])
                pr[new_d * n_ft + fid] += wr
        A_mix += rho * orient_entropy

    return phi_rho, pr, A_mix, A_internal


@_njit(cache=True)
def _eval_residual_implicit_oriented_nu(
    nu, W, phi, gamma_by_group, face_maps, delta,
    logZ_by_nocc, lm, n_ft
):
    Fd = W.shape[0]
    u = delta @ W
    X = np.empty(Fd, dtype=np.float64)
    hv = np.empty(Fd, dtype=np.float64)
    for s in range(Fd):
        xs = 1.0 / max(1.0 + u[s], 1e-12)
        xs = max(min(xs, 1.0), 1e-12)
        X[s] = xs
        hv[s] = np.log(max(xs, 1e-300)) - 0.5 * xs + 0.5

    _phi0, pr0, _mix0, _int0 = _implicit_oriented_stats_from_c_nu(
        hv, nu, gamma_by_group, face_maps, logZ_by_nocc, lm, n_ft
    )

    K = np.empty((Fd, Fd), dtype=np.float64)
    for s in range(Fd):
        xs2 = X[s] * X[s]
        for sp in range(Fd):
            K[s, sp] = xs2 * delta[s, sp] * pr0[sp]

    dAdX = np.empty(Fd, dtype=np.float64)
    for s in range(Fd):
        dAdX[s] = pr0[s] * (1.0 / max(X[s], 1e-12) - 0.5)

    Msys = np.empty((Fd, Fd), dtype=np.float64)
    for s in range(Fd):
        for sp in range(Fd):
            Msys[s, sp] = K[sp, s]
        Msys[s, s] += 1.0 + 1e-12

    alpha = np.linalg.solve(Msys, dAdX)
    beta = np.zeros(Fd, dtype=np.float64)
    for s in range(Fd):
        acc = 0.0
        for sp in range(Fd):
            acc += delta[sp, s] * alpha[sp] * X[sp] * X[sp]
        beta[s] = -acc

    c = np.empty(Fd, dtype=np.float64)
    for s in range(Fd):
        c[s] = hv[s] + beta[s] * X[s]

    phi_rho, pr, A_mix, A_internal = _implicit_oriented_stats_from_c_nu(
        c, nu, gamma_by_group, face_maps, logZ_by_nocc, lm, n_ft
    )

    R = np.empty(1 + Fd, dtype=np.float64)
    R[0] = phi_rho - phi
    for s in range(Fd):
        R[1 + s] = W[s] - X[s] * pr[s]

    A_assoc = 0.0
    for s in range(Fd):
        A_assoc += hv[s] * pr[s]
    f = (A_mix + A_internal + A_assoc) / 8.0
    return R, f


@_njit(cache=True)
def _find_initial_nu_implicit_oriented(
    phi_target, W, gamma_by_group, face_maps, delta,
    logZ_by_nocc, lm, n_ft
):
    Fd = W.shape[0]
    u = delta @ W
    hv = np.empty(Fd, dtype=np.float64)
    for s in range(Fd):
        xs = 1.0 / max(1.0 + u[s], 1e-12)
        xs = max(min(xs, 1.0), 1e-12)
        hv[s] = np.log(max(xs, 1e-300)) - 0.5 * xs + 0.5
    lo = -100.0
    hi = 100.0
    nu = 0.0
    for _ in range(140):
        phi_nu = _implicit_oriented_phi_from_c_nu(
            hv, nu, gamma_by_group, face_maps, logZ_by_nocc, lm, n_ft
        )
        if abs(phi_nu - phi_target) < 1e-12:
            break
        if phi_nu > phi_target:
            hi = nu
        else:
            lo = nu
        nu = 0.5 * (lo + hi)
    return nu


def _build_fd_jacobian_generic(residual_func, nu, W, phi, args, R, h_fd=1e-7):
    Fd = W.shape[0]
    n = 1 + Fd
    Jac = np.empty((n, n), dtype=np.float64)
    Rp, _ = residual_func(nu + h_fd, W, phi, *args)
    Jac[:, 0] = (Rp - R) / h_fd
    for j in range(Fd):
        Wp = W.copy()
        Wp[j] += h_fd
        Rp, _ = residual_func(nu, Wp, phi, *args)
        Jac[:, 1 + j] = (Rp - R) / h_fd
    Jac[np.diag_indices_from(Jac)] += 1e-12
    return Jac


def _solve_single_phi_broyden_generic(
    phi, nu0, W0, residual_func, initial_nu_func, args,
    *, tol=1e-8, max_iter=30, step_cap=10.0, h_fd=1e-7,
    prev_B=None, max_jac_rebuilds=1,
):
    Fd = W0.shape[0]
    n = 1 + Fd
    nu = float(nu0)
    W = W0.copy()
    R, f = residual_func(nu, W, phi, *args)
    rn = float(np.linalg.norm(R))
    if rn > 0.1:
        nu = float(initial_nu_func(phi, W, *args))
        R, f = residual_func(nu, W, phi, *args)
        rn = float(np.linalg.norm(R))

    best_x = np.empty(n, dtype=np.float64)
    best_x[0] = nu
    best_x[1:] = W.copy()
    best_rn = rn
    best_f = float(f) if np.isfinite(f) else np.nan
    if rn < tol:
        return {"x": best_x, "rn": best_rn, "f": best_f, "success": True, "B": prev_B}

    n_jac_builds = 0
    if prev_B is not None:
        B = prev_B.copy()
    else:
        Jac = _build_fd_jacobian_generic(residual_func, nu, W, phi, args, R, h_fd)
        try:
            B = np.linalg.inv(Jac)
        except np.linalg.LinAlgError:
            B = np.eye(n, dtype=np.float64)
        n_jac_builds = 1

    stall_count = 0
    for _it in range(max_iter):
        dx = -(B @ R)
        dx_norm = float(np.linalg.norm(dx))
        if dx_norm > step_cap:
            dx *= step_cap / dx_norm

        alpha_ls = 1.0
        R_new = None
        f_new = np.nan
        rn_new = rn
        for _ in range(4):
            nu_try = nu + alpha_ls * dx[0]
            W_try = W + alpha_ls * dx[1:]
            R_try, f_try = residual_func(nu_try, W_try, phi, *args)
            rn_try = float(np.linalg.norm(R_try))
            if rn_try < rn:
                R_new = R_try
                f_new = f_try
                rn_new = rn_try
                break
            alpha_ls *= 0.5

        if R_new is None:
            stall_count += 1
            if stall_count >= 3 and n_jac_builds < max_jac_rebuilds:
                Jac = _build_fd_jacobian_generic(residual_func, nu, W, phi, args, R, h_fd)
                try:
                    B = np.linalg.inv(Jac)
                except np.linalg.LinAlgError:
                    break
                n_jac_builds += 1
                stall_count = 0
            elif stall_count >= 3:
                break
            continue

        stall_count = 0
        step = alpha_ls * dx
        y = R_new - R
        By = B @ y
        denom = float(step @ By)
        if abs(denom) > 1e-30:
            B += np.outer(step - By, step @ B) / denom

        nu += alpha_ls * dx[0]
        W += alpha_ls * dx[1:]
        R = R_new
        f = f_new
        rn = rn_new
        if rn < best_rn:
            best_x[0] = nu
            best_x[1:] = W.copy()
            best_rn = rn
            best_f = float(f)
        if rn < tol:
            break

    return {"x": best_x, "rn": best_rn, "f": best_f,
            "success": best_rn <= tol, "B": B}


def _solve_free_energy_curve_nu_generic(
    phi_grid, Fd, residual_func, initial_nu_func, args,
    *, nu_init=0.0, W_init=None, tol=1e-8, accept_residual=1e-4,
    max_iter=30, max_jac_rebuilds=1,
):
    phi_grid = np.asarray(phi_grid, dtype=np.float64)
    fvals = np.full(len(phi_grid), np.nan, dtype=np.float64)
    nus = np.full(len(phi_grid), np.nan, dtype=np.float64)
    Ws_store = np.full((len(phi_grid), Fd), np.nan, dtype=np.float64)
    residuals = np.full(len(phi_grid), np.nan, dtype=np.float64)
    successes = np.zeros(len(phi_grid), dtype=bool)

    nu = float(nu_init)
    W = np.zeros(Fd, dtype=np.float64) if W_init is None else np.asarray(W_init, dtype=np.float64).copy()
    B = None
    for k, phi in enumerate(phi_grid):
        out = _solve_single_phi_broyden_generic(
            float(phi), nu, W, residual_func, initial_nu_func, args,
            tol=tol, max_iter=max_iter, prev_B=B,
            max_jac_rebuilds=max_jac_rebuilds,
        )
        residuals[k] = float(out["rn"])
        if np.isfinite(out["f"]) and out["rn"] <= accept_residual:
            nu = float(out["x"][0])
            W = out["x"][1:].copy()
            nus[k] = nu
            Ws_store[k, :] = W
            fvals[k] = float(out["f"])
            successes[k] = True
            B = out.get("B", None)
        else:
            B = None
    return {
        "phi": phi_grid,
        "f": fvals,
        "mus": nus,  # retained output key for backward compatibility; values are nu
        "nus": nus,
        "Ws": Ws_store,
        "residuals": residuals,
        "successes": successes,
        "mu_W": (float(nu), W.copy()),
        "nu_W": (float(nu), W.copy()),
    }


def solve_free_energy_curve_reduced_cube_sparse_nu(
    phi_grid, logZ_by_nocc, log_mult,
    pts, pss, m_patch, delta_small_np,
    *, nu_init=0.0, W_init=None, tol=1e-8,
    accept_residual=1e-4, max_iter=30, max_jac_rebuilds=1,
):
    logZ_by_nocc = np.ascontiguousarray(logZ_by_nocc, dtype=np.float64)
    log_mult = np.ascontiguousarray(log_mult, dtype=np.float64)
    pts = np.ascontiguousarray(pts)
    pss = np.ascontiguousarray(pss)
    m_patch = np.ascontiguousarray(m_patch)
    delta_small_np = np.ascontiguousarray(delta_small_np, dtype=np.float64)
    Fd = int(np.max(pss)) + 1
    args = (pts, pss, m_patch, delta_small_np, logZ_by_nocc, log_mult)
    return _solve_free_energy_curve_nu_generic(
        phi_grid, Fd,
        _eval_residual_reduced_cube_sparse_nu,
        _find_initial_nu_cube_sparse,
        args, nu_init=nu_init, W_init=W_init, tol=tol,
        accept_residual=accept_residual, max_iter=max_iter,
        max_jac_rebuilds=max_jac_rebuilds,
    )


def solve_free_energy_curve_reduced_cube_implicit_oriented_nu(
    phi_grid, logZ_by_nocc, log_mult,
    gamma_by_group, face_maps, delta_small_np, n_ft,
    *, nu_init=0.0, W_init=None, tol=1e-8,
    accept_residual=1e-4, max_iter=30, max_jac_rebuilds=1,
):
    logZ_by_nocc = np.ascontiguousarray(logZ_by_nocc, dtype=np.float64)
    log_mult = np.ascontiguousarray(log_mult, dtype=np.float64)
    gamma_by_group = np.ascontiguousarray(gamma_by_group, dtype=np.int64)
    face_maps = np.ascontiguousarray(face_maps, dtype=np.int64)
    delta_small_np = np.ascontiguousarray(delta_small_np, dtype=np.float64)
    Fd = int(delta_small_np.shape[0])
    args = (gamma_by_group, face_maps, delta_small_np,
            logZ_by_nocc, log_mult, int(n_ft))
    return _solve_free_energy_curve_nu_generic(
        phi_grid, Fd,
        _eval_residual_implicit_oriented_nu,
        _find_initial_nu_implicit_oriented,
        args, nu_init=nu_init, W_init=W_init, tol=tol,
        accept_residual=accept_residual, max_iter=max_iter,
        max_jac_rebuilds=max_jac_rebuilds,
    )

# ===================================================================
# State-point evaluation
# ===================================================================

def evaluate_state_point(
    eps_a, eps_c, *, patches, cache, ws, factor, phis,
    linear_term_mode="logZnu", registry_mode="boltzmann",
    association_mode="global_face",
    tol_solver=None, accept_residual=None, max_iter_solver=None, prev_mu_W=None,
    max_jac_rebuilds=1, fallback_newton=False,
    bethe_correction="none", bethe_strength=0.0, bethe_contact_factor=1.0,
    bethe_threshold=1e-12, bethe_compat_mode="attractive", bethe_orientation_mode="orbit_average", bethe_max_iter=500, bethe_tol=1e-10,
    bethe_sc_max_iter=20, bethe_sc_tol=1e-5, bethe_sc_damping=0.5,
):
    """Evaluate one grid point and return diagnostics for every point."""
    J = np.zeros((3, 3), dtype=np.float64)
    J[1, 1] = -float(eps_a)
    J[:-1, :-1] -= float(eps_c)

    mu_species = np.zeros(len(patches), dtype=np.float64)
    (
        eps_s, class_free_energy, logZ_internal, logZ_by_nocc,
        avg_Eeff, cube_to_species, m_patch, pts, pss, _cube_configs,
    ) = cubes_from_cache_fast(cache, J, mu_species, ws, registry_mode=registry_mode)

    A_row = np.sum(cube_to_species[:, :-1], axis=1) / 8.0

    mode = str(linear_term_mode).strip().lower().replace("-", "_")
    exact_internal_nu = False
    if mode in ("logznu", "logz_nu", "nu", "exact_nu", "class_free_energy_nu"):
        # Thermodynamically consistent mode: nu enters the internal class
        # partition function, so both log Z_g(nu) and the class composition
        # A_g(nu) are updated during every reduced residual evaluation.
        e_linear = np.ascontiguousarray(class_free_energy, dtype=np.float64)
        linear_term_mode_canonical = "logZnu"
        exact_internal_nu = True
    elif mode in ("avge", "avg_e", "avg", "average", "average_energy", "averaged_energy"):
        e_linear = np.ascontiguousarray(avg_Eeff, dtype=np.float64)
        linear_term_mode_canonical = "avgE"
    elif mode in ("logz", "free_energy", "class_free_energy", "classfe"):
        e_linear = np.ascontiguousarray(class_free_energy, dtype=np.float64)
        linear_term_mode_canonical = "logZ"
    else:
        raise ValueError(
            f"Unknown linear_term_mode={linear_term_mode!r}; "
            "use 'logZnu' (recommended), 'logZ', or 'avgE'."
        )

    reg_mode = str(registry_mode).strip().lower().replace("-", "_")
    if reg_mode in ("boltzmann", "boltz", "average", "avg", "boltzmann_average", "registry_average"):
        registry_mode_canonical = "boltzmann"
    elif reg_mode in ("min", "minimum", "min_energy", "best", "best_registry", "most_compatible"):
        registry_mode_canonical = "min"
    else:
        raise ValueError(f"Unknown registry_mode={registry_mode!r}; use 'boltzmann' or 'min'.")

    log_mult_np = np.zeros_like(e_linear)
    delta_np = (np.exp(-eps_s) - 1.0) / float(factor)

    association_mode_canonical = str(association_mode).strip().lower().replace("-", "_")
    assoc_gamma_status = "global_face"
    implicit_oriented_inputs = None
    if association_mode_canonical in ("global", "global_face", "standard", "saftp", "original"):
        association_mode_canonical = "global_face"
    elif association_mode_canonical in ("directed_face", "directed_face_state", "state", "state_assoc", "x_gs", "x_gd", "direction_a"):
        pts, pss, m_patch, delta_np, assoc_gamma_status = _build_directed_face_assoc_arrays(cache, ws, eps_s, delta_np)
        association_mode_canonical = "directed_face_state"
    elif association_mode_canonical in ("oriented_directed_face", "oriented_directed_face_state", "oriented_state", "full_oriented", "oriented"):
        # Low-memory exact implementation of the same explicit oriented-state closure.
        # It sums over the 24 orientations inside the residual instead of materializing
        # gamma_or, pts_dir, pss_dir, m_dir, and repeated A/e/log-multiplicity arrays.
        gamma_impl, face_maps_impl, delta_np, n_orient, assoc_gamma_status = _build_implicit_oriented_directed_face_assoc_inputs(cache, ws, eps_s, delta_np)
        log_mult_np = np.zeros_like(e_linear)
        implicit_oriented_inputs = (gamma_impl, face_maps_impl, int(eps_s.shape[0]))
        association_mode_canonical = "oriented_directed_face_state"
    elif association_mode_canonical in ("explicit_oriented_directed_face", "explicit_oriented_directed_face_state", "oriented_directed_face_state_explicit"):
        # Debug/reference path: old memory-heavy materialized (g,R) implementation.
        pts, pss, m_patch, delta_np, lm_or, n_orient, assoc_gamma_status = _build_oriented_directed_face_assoc_arrays(cache, ws, eps_s, delta_np)
        A_row = np.repeat(A_row, int(n_orient))
        e_linear = np.repeat(e_linear, int(n_orient))
        logZ_by_nocc = np.repeat(logZ_by_nocc, int(n_orient), axis=0)
        log_mult_np = lm_or
        association_mode_canonical = "explicit_oriented_directed_face_state"
    else:
        raise ValueError(f"Unknown association_mode={association_mode!r}; use global_face, directed_face_state, or oriented_directed_face_state.")

    if association_mode_canonical in ("oriented_directed_face_state", "explicit_oriented_directed_face_state") and str(bethe_correction).strip().lower() not in ("none", "off", "0", "false"):
        raise ValueError(
            "oriented_directed_face_state currently supports BETHE_CORRECTION=none only. "
            "Boundary Bethe needs an oriented gamma map with P_or states; run the association correction first."
        )

    if exact_internal_nu and str(bethe_correction).strip().lower() not in ("none", "off", "0", "false"):
        raise ValueError(
            "linear_term_mode=logZnu currently supports BETHE_CORRECTION=none only. "
            "The Bethe post-processing path assumes fixed class compositions."
        )

    phis_requested = np.asarray(phis, dtype=np.float64)
    if exact_internal_nu:
        occ_support = np.where(np.any(np.isfinite(logZ_by_nocc), axis=0))[0].astype(np.float64) / 8.0
        if occ_support.size == 0:
            phi_feasible_min = np.nan
            phi_feasible_max = np.nan
            phi_feasible_mask = np.zeros_like(phis_requested, dtype=bool)
        else:
            phi_feasible_min = float(np.min(occ_support))
            phi_feasible_max = float(np.max(occ_support))
            phi_feasible_mask = (phis_requested >= phi_feasible_min - 1e-12) & (phis_requested <= phi_feasible_max + 1e-12)
        phis_solve = phis_requested[phi_feasible_mask]
    else:
        phis_solve, phi_feasible_mask, phi_feasible_min, phi_feasible_max = filter_feasible_phi_grid(
            phis_requested, A_row, atol=1e-12
        )
    n_phi_requested = int(len(phis_requested))
    n_phi_invalid = int(n_phi_requested - len(phis_solve))
    if exact_internal_nu:
        A_diag_min = float(phi_feasible_min)
        A_diag_max = float(phi_feasible_max)
    else:
        A_diag_min = float(np.nanmin(A_row))
        A_diag_max = float(np.nanmax(A_row))

    out_base = {
        "phi_requested_min": float(np.nanmin(phis_requested)) if phis_requested.size else np.nan,
        "phi_requested_max": float(np.nanmax(phis_requested)) if phis_requested.size else np.nan,
        "phi_feasible_min": float(phi_feasible_min),
        "phi_feasible_max": float(phi_feasible_max),
        "n_phi_requested": n_phi_requested,
        "n_phi_invalid": n_phi_invalid,
    }

    if len(phis_solve) < 5:
        out = {
            "eps_a": float(eps_a), "eps_c": float(eps_c),
            "event_detected": False, "binodal_detected": False, "spinodal_detected": False,
            "phi1": np.nan, "phi2": np.nan, "mu": np.nan,
            "barrier": np.nan, "below_min": np.nan, "n_points": 0,
            "spinodal_phis": [], "spinodal_n_crossings": 0,
            "spinodal_min_fpp": np.nan, "spinodal_phi_at_min_fpp": np.nan,
            "spinodal_tol": 1e-6,
            "n_phi": int(len(phis_solve)), "n_valid_phi": 0, "n_success_phi": 0,
            "success_fraction": 0.0,
            "max_residual": np.nan, "mean_residual": np.nan,
            "median_residual": np.nan, "phi_at_max_residual": np.nan,
            "min_f": np.nan, "max_f": np.nan,
            "min_mu": np.nan, "max_mu": np.nan,
            "A_min": A_diag_min, "A_max": A_diag_max,
            "linear_term_mode": linear_term_mode_canonical,
            "registry_mode": registry_mode_canonical,
            "association_mode": association_mode_canonical,
            "association_gamma_status": assoc_gamma_status,
            "e_linear_min": float(np.nanmin(e_linear)),
            "e_linear_max": float(np.nanmax(e_linear)),
            "avg_Eeff_min": float(np.nanmin(avg_Eeff)),
            "avg_Eeff_max": float(np.nanmax(avg_Eeff)),
            "class_free_energy_min": float(np.nanmin(class_free_energy)),
            "class_free_energy_max": float(np.nanmax(class_free_energy)),
            "logZ_internal_min": float(np.nanmin(logZ_internal)),
            "logZ_internal_max": float(np.nanmax(logZ_internal)),
            "delta_min": float(np.nanmin(delta_np)),
            "delta_max": float(np.nanmax(delta_np)),
            "delta_n_nonzero": int(np.count_nonzero(np.abs(delta_np) > 0.0)),
            "bethe_correction": str(bethe_correction),
            "bethe_strength": float(bethe_strength),
            "bethe_contact_factor": float(bethe_contact_factor),
            "bethe_threshold": float(bethe_threshold),
            "bethe_compat_mode": str(bethe_compat_mode),
            "bethe_orientation_mode": str(bethe_orientation_mode),
            "bethe_gamma_status": "not_evaluated",
            "bethe_raw_median": np.nan,
            "bethe_corr_median": np.nan,
            "bethe_allowed_prob_median": np.nan,
            "bethe_ok_fraction": 0.0,
            "diagnostic_status": "too_few_feasible_phi_after_filter",
        }
        out.update(out_base)
        return out, prev_mu_W

    mu0, W0 = prev_mu_W if prev_mu_W is not None else (0.0, None)
    bethe_mode = str(bethe_correction).strip().lower().replace("-", "_")
    if bethe_mode in ("selfconsistent_boundary_bethe", "self_consistent_boundary_bethe", "sc_boundary_bethe", "cavity_bethe"):
        curve = solve_free_energy_curve_reduced_cube_sparse_selfconsistent_bethe(
            phis_solve, A_row, e_linear, log_mult_np,
            pts, pss, m_patch, delta_np,
            cache=cache, ws=ws, eps_small=eps_s,
            compat_mode=bethe_compat_mode,
            orientation_mode=bethe_orientation_mode,
            threshold=bethe_threshold,
            bethe_max_iter=bethe_max_iter,
            bethe_tol=bethe_tol,
            sc_max_iter=bethe_sc_max_iter,
            sc_tol=bethe_sc_tol,
            sc_damping=bethe_sc_damping,
            mu_init=mu0, W_init=W0,
            tol=tol_solver, accept_residual=accept_residual,
            max_iter=max_iter_solver,
            fallback_newton=fallback_newton,
        )
        next_mu_W = curve["mu_W"]
        phi_arr = np.asarray(curve["phi"], dtype=np.float64)
        f_arr_base = np.asarray(curve.get("f_base", curve["f"]), dtype=np.float64)
        f_arr = np.asarray(curve["f"], dtype=np.float64)
        residuals = np.asarray(curve["residuals"], dtype=np.float64)
        successes = np.asarray(curve["successes"], dtype=bool)
        mus = np.asarray(curve["mus"], dtype=np.float64)
        bethe_raw = np.asarray(curve.get("bethe_raw", np.full_like(f_arr, np.nan)), dtype=np.float64)
        bethe_corr = np.asarray(curve.get("bethe_corr", np.full_like(f_arr, np.nan)), dtype=np.float64)
        bethe_prob = np.asarray(curve.get("bethe_prob", np.full_like(f_arr, np.nan)), dtype=np.float64)
        bethe_ok = np.asarray(curve.get("bethe_ok", np.zeros_like(successes)), dtype=bool)
        bethe_iters = np.asarray(curve.get("bethe_ipf_iters", np.zeros_like(successes, dtype=np.int32)), dtype=np.int32)
        bethe_gamma_status = str(curve.get("bethe_gamma_status", "not_used"))
    else:
        if implicit_oriented_inputs is not None:
            gamma_impl, face_maps_impl, n_ft_impl = implicit_oriented_inputs
            if exact_internal_nu:
                curve = solve_free_energy_curve_reduced_cube_implicit_oriented_nu(
                    phis_solve, logZ_by_nocc, log_mult_np,
                    gamma_impl, face_maps_impl, delta_np, n_ft_impl,
                    nu_init=mu0, W_init=W0,
                    tol=tol_solver, accept_residual=accept_residual,
                    max_iter=max_iter_solver,
                    max_jac_rebuilds=max_jac_rebuilds,
                )
            else:
                curve = solve_free_energy_curve_reduced_cube_implicit_oriented(
                    phis_solve, A_row, e_linear, log_mult_np,
                    gamma_impl, face_maps_impl, delta_np, n_ft_impl,
                    mu_init=mu0, W_init=W0,
                    tol=tol_solver, accept_residual=accept_residual,
                    max_iter=max_iter_solver,
                    max_jac_rebuilds=max_jac_rebuilds,
                    fallback_newton=fallback_newton,
                )
            next_mu_W = curve["mu_W"]

            phi_arr = np.asarray(curve["phi"], dtype=np.float64)
            f_arr_base = np.asarray(curve["f"], dtype=np.float64)
            residuals = np.asarray(curve["residuals"], dtype=np.float64)
            successes = np.asarray(curve["successes"], dtype=bool)
            mus = np.asarray(curve["mus"], dtype=np.float64)

            bethe_corr = np.zeros_like(f_arr_base)
            bethe_raw = np.zeros_like(f_arr_base)
            bethe_prob = np.ones_like(f_arr_base)
            bethe_ok = np.ones_like(successes, dtype=bool)
            bethe_iters = np.zeros_like(successes, dtype=np.int32)
            bethe_gamma_status = "not_used_oriented_implicit"
            f_arr = f_arr_base
        else:
            if exact_internal_nu:
                curve = solve_free_energy_curve_reduced_cube_sparse_nu(
                    phis_solve, logZ_by_nocc, log_mult_np,
                    pts, pss, m_patch, delta_np,
                    nu_init=mu0, W_init=W0,
                    tol=tol_solver, accept_residual=accept_residual,
                    max_iter=max_iter_solver,
                    max_jac_rebuilds=max_jac_rebuilds,
                )
            else:
                curve = solve_free_energy_curve_reduced_cube_sparse(
                    phis_solve, A_row, e_linear, log_mult_np,
                    pts, pss, m_patch, delta_np,
                    mu_init=mu0, W_init=W0,
                    tol=tol_solver, accept_residual=accept_residual,
                    max_iter=max_iter_solver,
                    max_jac_rebuilds=max_jac_rebuilds,
                    fallback_newton=fallback_newton,
                )
            next_mu_W = curve["mu_W"]

            phi_arr = np.asarray(curve["phi"], dtype=np.float64)
            f_arr_base = np.asarray(curve["f"], dtype=np.float64)
            residuals = np.asarray(curve["residuals"], dtype=np.float64)
            successes = np.asarray(curve["successes"], dtype=bool)
            mus = np.asarray(curve["mus"], dtype=np.float64)

            if exact_internal_nu:
                bethe_corr = np.zeros_like(f_arr_base)
                bethe_raw = np.zeros_like(f_arr_base)
                bethe_prob = np.ones_like(f_arr_base)
                bethe_ok = np.ones_like(successes, dtype=bool)
                bethe_iters = np.zeros_like(successes, dtype=np.int32)
                bethe_gamma_status = "not_used_logZnu"
                f_arr = f_arr_base
            else:
                bethe_corr, bethe_raw, bethe_prob, bethe_ok, bethe_iters, bethe_gamma_status = compute_bethe_curve_correction(
                    curve, A_row, e_linear, log_mult_np,
                    pts, pss, m_patch, delta_np, eps_s,
                    cache=cache, ws=ws,
                    correction=bethe_correction,
                    strength=bethe_strength,
                    contact_factor=bethe_contact_factor,
                    threshold=bethe_threshold,
                    compat_mode=bethe_compat_mode,
                    orientation_mode=bethe_orientation_mode,
                    max_iter=bethe_max_iter,
                    tol=bethe_tol,
                )
                f_arr = f_arr_base + bethe_corr

    valid = np.isfinite(f_arr)
    n_phi = int(len(phi_arr))
    n_valid = int(np.count_nonzero(valid))
    n_success = int(np.count_nonzero(successes))
    finite_res = residuals[np.isfinite(residuals)]
    max_res = float(np.nanmax(finite_res)) if finite_res.size else np.nan
    mean_res = float(np.nanmean(finite_res)) if finite_res.size else np.nan
    med_res = float(np.nanmedian(finite_res)) if finite_res.size else np.nan
    max_res_idx = int(np.nanargmax(residuals)) if np.any(np.isfinite(residuals)) else -1
    phi_at_max_res = float(phi_arr[max_res_idx]) if max_res_idx >= 0 else np.nan

    out = {
        "eps_a": float(eps_a), "eps_c": float(eps_c),
        "event_detected": False, "binodal_detected": False, "spinodal_detected": False,
        "phi1": np.nan, "phi2": np.nan, "mu": np.nan,
        "barrier": np.nan, "below_min": np.nan, "n_points": 0,
        "spinodal_phis": [], "spinodal_n_crossings": 0,
        "spinodal_min_fpp": np.nan, "spinodal_phi_at_min_fpp": np.nan,
        "spinodal_tol": 1e-6,
        "n_phi": n_phi, "n_valid_phi": n_valid, "n_success_phi": n_success,
        "success_fraction": float(n_success / max(n_phi, 1)),
        "max_residual": max_res, "mean_residual": mean_res,
        "median_residual": med_res, "phi_at_max_residual": phi_at_max_res,
        "min_f": float(np.nanmin(f_arr[valid])) if np.any(valid) else np.nan,
        "max_f": float(np.nanmax(f_arr[valid])) if np.any(valid) else np.nan,
        "min_mu": float(np.nanmin(mus[np.isfinite(mus)])) if np.any(np.isfinite(mus)) else np.nan,
        "max_mu": float(np.nanmax(mus[np.isfinite(mus)])) if np.any(np.isfinite(mus)) else np.nan,
        "A_min": A_diag_min, "A_max": A_diag_max,
        "linear_term_mode": linear_term_mode_canonical,
        "registry_mode": registry_mode_canonical,
        "association_mode": association_mode_canonical,
        "association_gamma_status": assoc_gamma_status,
        "e_linear_min": float(np.nanmin(e_linear)),
        "e_linear_max": float(np.nanmax(e_linear)),
        "avg_Eeff_min": float(np.nanmin(avg_Eeff)),
        "avg_Eeff_max": float(np.nanmax(avg_Eeff)),
        "class_free_energy_min": float(np.nanmin(class_free_energy)),
        "class_free_energy_max": float(np.nanmax(class_free_energy)),
        "logZ_internal_min": float(np.nanmin(logZ_internal)),
        "logZ_internal_max": float(np.nanmax(logZ_internal)),
        "delta_min": float(np.nanmin(delta_np)),
        "delta_max": float(np.nanmax(delta_np)),
        "delta_n_nonzero": int(np.count_nonzero(np.abs(delta_np) > 0.0)),
        "bethe_correction": str(bethe_correction),
        "bethe_strength": float(bethe_strength),
        "bethe_contact_factor": float(bethe_contact_factor),
        "bethe_threshold": float(bethe_threshold),
        "bethe_compat_mode": str(bethe_compat_mode),
        "bethe_orientation_mode": str(bethe_orientation_mode),
        "bethe_gamma_status": str(bethe_gamma_status),
        "bethe_raw_median": float(np.nanmedian(bethe_raw[np.isfinite(bethe_raw)])) if np.any(np.isfinite(bethe_raw)) else np.nan,
        "bethe_corr_median": float(np.nanmedian(bethe_corr[np.isfinite(bethe_corr)])) if np.any(np.isfinite(bethe_corr)) else np.nan,
        "bethe_allowed_prob_median": float(np.nanmedian(bethe_prob[np.isfinite(bethe_prob)])) if np.any(np.isfinite(bethe_prob)) else np.nan,
        "bethe_ok_fraction": float(np.count_nonzero(bethe_ok) / max(len(bethe_ok), 1)),
    }
    out.update(out_base)

    phi_v, f_v = phi_arr[valid], f_arr[valid]
    if len(phi_v) < 5:
        out["diagnostic_status"] = "too_few_valid_phi"
        return out, next_mu_W

    # Curvature diagnostics are unreliable at the phi-grid endpoints because
    # LOESS uses one-sided neighborhoods there and the ideal entropy has very
    # large endpoint curvature.  Do not let an endpoint artifact declare a
    # spinodal.  The default margin removes roughly 5% of the phi window on each
    # side, but always keeps enough points for the envelope/binodal test.
    phi_span = float(np.nanmax(phi_v) - np.nanmin(phi_v)) if len(phi_v) else 0.0
    phi_margin_abs = 0.05 * phi_span
    edge_mask = (phi_v >= float(np.nanmin(phi_v)) + phi_margin_abs) & (phi_v <= float(np.nanmax(phi_v)) - phi_margin_abs)
    if np.count_nonzero(edge_mask) >= 9:
        phi_inner = phi_v[edge_mask]
        f_inner = f_v[edge_mask]
    else:
        margin = min(max(3, len(phi_v) // 20), len(phi_v) // 4)
        phi_inner = phi_v[margin:-margin] if margin > 0 else phi_v
        f_inner = f_v[margin:-margin] if margin > 0 else f_v
    if len(phi_inner) < 5:
        out["diagnostic_status"] = "too_few_inner_phi"
        return out, next_mu_W

    f_smooth, _, fpp = loess_derivs(phi_inner, f_inner)
    spinodal_tol = 1e-6
    min_fpp = float(np.nanmin(fpp))
    max_fpp = float(np.nanmax(fpp))
    argmin_fpp = int(np.nanargmin(fpp))
    sp = zero_crossings_linear(phi_inner, -fpp, tol=spinodal_tol)
    # A physical spinodal should be an interior negative-curvature interval,
    # bracketed by positive curvature somewhere on the scanned branch.  A mere
    # negative minimum, especially a globally concave LOESS fit, is diagnostic
    # failure rather than a stable phase-boundary signal.
    spinodal_detected = bool((min_fpp < -spinodal_tol) and (max_fpp > spinodal_tol) and (len(sp) > 0))
    if (min_fpp < -spinodal_tol) and not spinodal_detected:
        out["diagnostic_status"] = "negative_curvature_unbracketed_not_counted_as_spinodal"

    res = extract_binodals_from_convex_envelope(phi_inner, f_smooth, coexist_tol=1e-5, min_gap_points=6)
    binodal = best_binodal_segment(res, prefer="largest_barrier")
    binodal_detected = bool(binodal is not None and binodal["barrier"] > 1e-4)
    if binodal_detected:
        out.update(dict(binodal))

    if out.get("diagnostic_status", "") not in ("negative_curvature_unbracketed_not_counted_as_spinodal",):
        out["diagnostic_status"] = "ok"
    out["binodal_detected"] = binodal_detected
    out["spinodal_detected"] = spinodal_detected
    out["event_detected"] = bool(spinodal_detected or binodal_detected)
    out["spinodal_phis"] = [float(x) for x in sp]
    out["spinodal_n_crossings"] = int(len(sp))
    out["spinodal_min_fpp"] = min_fpp
    out["spinodal_phi_at_min_fpp"] = float(phi_inner[argmin_fpp])
    out["fpp_at_phi_min"] = float(fpp[0])
    out["fpp_at_phi_max"] = float(fpp[-1])
    out["fpp_median"] = float(np.nanmedian(fpp))
    out["fpp_max"] = max_fpp
    out["fpp_min_abs"] = float(np.nanmin(np.abs(fpp)))
    out["phi_at_min_abs_fpp"] = float(phi_inner[int(np.nanargmin(np.abs(fpp)))])
    return out, next_mu_W


# ===================================================================
# CLI
# ===================================================================

def cmd_prepare_mmap(args):
    prepare_mmap_cache(args.cache_path, args.mmap_dir, verbose=True)


def cmd_run_shard(args):
    # Pin glibc mmap threshold BEFORE any heavy allocation
    _pin_malloc_mmap_threshold()

    patches = load_patches_npy(args.patches)
    eps_as, eps_cs = grid_from_args(args)
    phis = np.linspace(args.phi_min, args.phi_max, args.n_phis)

    mmap_dir = Path(args.mmap_dir) if args.mmap_dir else Path(args.cache_path).with_suffix(".mmap")
    if not (mmap_dir / ".ready").exists():
        raise FileNotFoundError(
            f"mmap cache not found at {mmap_dir}. Run prepare-mmap first.")

    log(f"loading mmap cache from {mmap_dir}")
    cache = load_cache_mmap(str(mmap_dir))

    # Pre-allocate workspace (reused every cubes_from_cache call)
    ws = CubeWorkspace(cache, use_boundary_quotient=args.use_boundary_quotient)
    log(
        f"workspace allocated: n_groups={ws.n_groups}, n_ft={ws.n_ft}, "
        f"factor={args.factor}, linear_term_mode={args.linear_term_mode}, "
        f"registry_mode={args.registry_mode}, "
        f"association_mode={args.association_mode}, "
        f"use_boundary_quotient={args.use_boundary_quotient}, "
        f"cfg_includes_full_orbit={ws.cfg_includes_full_orbit}, "
        f"boundary_orbit_mult[min,max,mean]={ws.boundary_orbit_mult_stats}, "
        f"boundary_patch_mode={cache.get('boundary_patch_mode', 'unknown')}, "
        f"boundary_sparse_entries={cache.get('boundary_sparse_entries', 'NA')}, "
        f"patches_per_group_mean={cache.get('patches_per_group_mean', 'NA')}, "
        f"m_patch_sum[min,max]=({cache.get('m_patch_sum_min', 'NA')},{cache.get('m_patch_sum_max', 'NA')})"
    )


    # Report memory after setup
    try:
        with open(f"/proc/{os.getpid()}/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    log(f"RSS after setup: {line.strip()}")
                    break
    except Exception:
        pass

    n_total = len(eps_as) * len(eps_cs)
    start, stop = contiguous_shard_bounds(n_total, args.shard_id, args.n_shards)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_npz = out_dir / f"shard_{args.shard_id:04d}.npz"
    out_json = out_dir / f"shard_{args.shard_id:04d}.json"

    log(f"shard {args.shard_id}/{args.n_shards}: range=[{start}, {stop}) count={stop-start}")

    t0 = time.perf_counter()
    prev_mu_W = None
    all_results = []
    found = []

    for processed, flat_idx in enumerate(range(start, stop), start=1):
        j = flat_idx // len(eps_as)
        i = flat_idx % len(eps_as)
        eps_a = float(eps_as[i])
        eps_c = float(eps_cs[j])

        point_t0 = time.perf_counter()
        result, prev_mu_W = evaluate_state_point(
            eps_a, eps_c,
            patches=patches, cache=cache, ws=ws, factor=args.factor,
            phis=phis, linear_term_mode=args.linear_term_mode, registry_mode=args.registry_mode,
            association_mode=args.association_mode,
            tol_solver=args.tol_solver,
            accept_residual=args.accept_residual,
            max_iter_solver=args.max_iter_solver,
            prev_mu_W=prev_mu_W,
            max_jac_rebuilds=args.max_jac_rebuilds,
            fallback_newton=args.fallback_newton,
            bethe_correction=args.bethe_correction,
            bethe_strength=args.bethe_strength,
            bethe_contact_factor=args.bethe_contact_factor,
            bethe_threshold=args.bethe_threshold,
            bethe_compat_mode=args.bethe_compat_mode,
            bethe_orientation_mode=args.bethe_orientation_mode,
            bethe_max_iter=args.bethe_max_iter,
            bethe_tol=args.bethe_tol,
            bethe_sc_max_iter=args.bethe_sc_max_iter,
            bethe_sc_tol=args.bethe_sc_tol,
            bethe_sc_damping=args.bethe_sc_damping,
        )
        dt = time.perf_counter() - point_t0

        # Release fragmented memory EVERY state point — the Broyden solver
        # creates millions of ~2 MB temporaries via Numba NRT → malloc
        _release_memory()

        if processed == 1 or processed % max(1, min(10, (stop - start) // 4)) == 0:
            rss_mb = "?"
            try:
                with open(f"/proc/{os.getpid()}/status") as f:
                    for line in f:
                        if line.startswith("VmRSS:"):
                            rss_mb = line.split()[1]
                            break
            except Exception:
                pass
            log(
                f"shard {args.shard_id}: {processed}/{stop-start} "
                f"eps_a={eps_a:.3f} eps_c={eps_c:.3f} dt={dt:.1f}s "
                f"found={'Y' if result else 'N'} RSS={rss_mb}kB"
            )

        if result is not None:
            all_results.append(result)
            if result.get("event_detected", False):
                found.append(result)
            log(
                f"POINT shard={args.shard_id} idx={processed}/{stop-start} "
                f"eps_a={eps_a:.6g} eps_c={eps_c:.6g} dt={dt:.2f}s "
                f"event={int(result.get('event_detected', False))} "
                f"spin={int(result.get('spinodal_detected', False))} "
                f"bin={int(result.get('binodal_detected', False))} "
                f"min_fpp={result.get('spinodal_min_fpp', np.nan):.6e} "
                f"phi_min_fpp={result.get('spinodal_phi_at_min_fpp', np.nan):.6g} "
                f"n_valid={result.get('n_valid_phi', -1)}/{result.get('n_phi', -1)} "
                f"n_invalid_phi={result.get('n_phi_invalid', -1)} "
                f"phi_feasible=[{result.get('phi_feasible_min', np.nan):.6g},{result.get('phi_feasible_max', np.nan):.6g}] "
                f"n_success={result.get('n_success_phi', -1)} "
                f"max_res={result.get('max_residual', np.nan):.3e} "
                f"mean_res={result.get('mean_residual', np.nan):.3e} "
                f"mode={result.get('linear_term_mode', 'NA')} "
                f"e_lin=[{result.get('e_linear_min', np.nan):.3e},{result.get('e_linear_max', np.nan):.3e}] "
                f"barrier={result.get('barrier', np.nan):.3e} "
                f"bethe={result.get('bethe_correction', 'none')} "
                f"bmode={result.get('bethe_compat_mode', 'NA')} "
                f"borient={result.get('bethe_orientation_mode', 'NA')} "
                f"bgamma={result.get('bethe_gamma_status', 'NA')} "
                f"braw_med={result.get('bethe_raw_median', np.nan):.3e} "
                f"bcorr_med={result.get('bethe_corr_median', np.nan):.3e} "
                f"bp_med={result.get('bethe_allowed_prob_median', np.nan):.3e} "
                f"status={result.get('diagnostic_status', 'NA')}"
            )

    if found:
        points = np.array([[d["eps_a"], d["eps_c"]] for d in found], dtype=np.float64)
        binodal_phi1 = np.array([d["phi1"] for d in found], dtype=np.float64)
        binodal_phi2 = np.array([d["phi2"] for d in found], dtype=np.float64)
        binodal_barrier = np.array([d["barrier"] for d in found], dtype=np.float64)
        binodal_eps_a = np.array([d["eps_a"] for d in found], dtype=np.float64)
        binodal_eps_c = np.array([d["eps_c"] for d in found], dtype=np.float64)
        binodal_detected = np.array([d["binodal_detected"] for d in found], dtype=bool)
        spinodal_detected = np.array([d["spinodal_detected"] for d in found], dtype=bool)
        spinodal_n_crossings = np.array([d["spinodal_n_crossings"] for d in found], dtype=np.int64)
        spinodal_min_fpp = np.array([d["spinodal_min_fpp"] for d in found], dtype=np.float64)
        spinodal_phi_at_min_fpp = np.array([d["spinodal_phi_at_min_fpp"] for d in found], dtype=np.float64)
        max_crossings = max([len(d["spinodal_phis"]) for d in found] + [0])
        spinodal_phis = np.full((len(found), max_crossings), np.nan, dtype=np.float64)
        for row, d in enumerate(found):
            vals = np.asarray(d["spinodal_phis"], dtype=np.float64)
            spinodal_phis[row, :len(vals)] = vals
    else:
        points = np.empty((0, 2), dtype=np.float64)
        binodal_phi1 = np.array([], dtype=np.float64)
        binodal_phi2 = np.array([], dtype=np.float64)
        binodal_barrier = np.array([], dtype=np.float64)
        binodal_eps_a = np.array([], dtype=np.float64)
        binodal_eps_c = np.array([], dtype=np.float64)
        binodal_detected = np.array([], dtype=bool)
        spinodal_detected = np.array([], dtype=bool)
        spinodal_n_crossings = np.array([], dtype=np.int64)
        spinodal_min_fpp = np.array([], dtype=np.float64)
        spinodal_phi_at_min_fpp = np.array([], dtype=np.float64)
        spinodal_phis = np.empty((0, 0), dtype=np.float64)

    # Full-grid diagnostics, including non-detected points.
    diag_keys_float = [
        "spinodal_min_fpp", "spinodal_phi_at_min_fpp", "fpp_min_abs", "phi_at_min_abs_fpp", "fpp_max",
        "success_fraction", "max_residual", "mean_residual", "median_residual",
        "phi_at_max_residual", "barrier", "phi1", "phi2", "delta_min", "delta_max",
        "A_min", "A_max", "min_f", "max_f", "min_mu", "max_mu",
        "e_linear_min", "e_linear_max", "avg_Eeff_min", "avg_Eeff_max",
        "class_free_energy_min", "class_free_energy_max", "logZ_internal_min", "logZ_internal_max",
        "bethe_raw_median", "bethe_corr_median", "bethe_allowed_prob_median",
        "bethe_ok_fraction",
    ]
    if all_results:
        diag_eps_a = np.array([d["eps_a"] for d in all_results], dtype=np.float64)
        diag_eps_c = np.array([d["eps_c"] for d in all_results], dtype=np.float64)
        diag_event_detected = np.array([d.get("event_detected", False) for d in all_results], dtype=bool)
        diag_spinodal_detected = np.array([d.get("spinodal_detected", False) for d in all_results], dtype=bool)
        diag_binodal_detected = np.array([d.get("binodal_detected", False) for d in all_results], dtype=bool)
        diag_n_phi = np.array([d.get("n_phi", 0) for d in all_results], dtype=np.int64)
        diag_n_valid_phi = np.array([d.get("n_valid_phi", 0) for d in all_results], dtype=np.int64)
        diag_n_success_phi = np.array([d.get("n_success_phi", 0) for d in all_results], dtype=np.int64)
        diag_n_phi_requested = np.array([d.get("n_phi_requested", d.get("n_phi", 0)) for d in all_results], dtype=np.int64)
        diag_n_phi_invalid = np.array([d.get("n_phi_invalid", 0) for d in all_results], dtype=np.int64)
        diag_status = np.array([str(d.get("diagnostic_status", "NA")) for d in all_results])
        diag_linear_term_mode = np.array([str(d.get("linear_term_mode", "NA")) for d in all_results])
        diag_registry_mode = np.array([str(d.get("registry_mode", "NA")) for d in all_results])
        diag_bethe_correction = np.array([str(d.get("bethe_correction", "none")) for d in all_results])
        diag_bethe_compat_mode = np.array([str(d.get("bethe_compat_mode", "NA")) for d in all_results])
        diag_bethe_orientation_mode = np.array([str(d.get("bethe_orientation_mode", "NA")) for d in all_results])
        diag_bethe_gamma_status = np.array([str(d.get("bethe_gamma_status", "NA")) for d in all_results])
        diag_float = {k: np.array([d.get(k, np.nan) for d in all_results], dtype=np.float64) for k in diag_keys_float}
    else:
        diag_eps_a = np.array([], dtype=np.float64)
        diag_eps_c = np.array([], dtype=np.float64)
        diag_event_detected = np.array([], dtype=bool)
        diag_spinodal_detected = np.array([], dtype=bool)
        diag_binodal_detected = np.array([], dtype=bool)
        diag_n_phi = np.array([], dtype=np.int64)
        diag_n_valid_phi = np.array([], dtype=np.int64)
        diag_n_success_phi = np.array([], dtype=np.int64)
        diag_n_phi_requested = np.array([], dtype=np.int64)
        diag_n_phi_invalid = np.array([], dtype=np.int64)
        diag_status = np.array([], dtype=str)
        diag_linear_term_mode = np.array([], dtype=str)
        diag_registry_mode = np.array([], dtype=str)
        diag_bethe_correction = np.array([], dtype=str)
        diag_bethe_compat_mode = np.array([], dtype=str)
        diag_bethe_orientation_mode = np.array([], dtype=str)
        diag_bethe_gamma_status = np.array([], dtype=str)
        diag_float = {k: np.array([], dtype=np.float64) for k in diag_keys_float}

    np.savez_compressed(
        out_npz,
        points=points, binodal_phi1=binodal_phi1,
        binodal_phi2=binodal_phi2, binodal_barrier=binodal_barrier,
        binodal_eps_a=binodal_eps_a, binodal_eps_c=binodal_eps_c,
        binodal_detected=binodal_detected,
        spinodal_detected=spinodal_detected,
        spinodal_n_crossings=spinodal_n_crossings,
        spinodal_min_fpp=spinodal_min_fpp,
        spinodal_phi_at_min_fpp=spinodal_phi_at_min_fpp,
        spinodal_phis=spinodal_phis,
        diag_eps_a=diag_eps_a, diag_eps_c=diag_eps_c,
        diag_event_detected=diag_event_detected,
        diag_spinodal_detected=diag_spinodal_detected,
        diag_binodal_detected=diag_binodal_detected,
        diag_n_phi=diag_n_phi,
        diag_n_valid_phi=diag_n_valid_phi,
        diag_n_success_phi=diag_n_success_phi,
        diag_n_phi_requested=diag_n_phi_requested,
        diag_n_phi_invalid=diag_n_phi_invalid,
        diag_status=diag_status,
        diag_linear_term_mode=diag_linear_term_mode,
        diag_registry_mode=diag_registry_mode,
        diag_bethe_correction=diag_bethe_correction,
        diag_bethe_compat_mode=diag_bethe_compat_mode,
        diag_bethe_orientation_mode=diag_bethe_orientation_mode,
        diag_bethe_gamma_status=diag_bethe_gamma_status,
        diag_spinodal_min_fpp=diag_float["spinodal_min_fpp"],
        diag_spinodal_phi_at_min_fpp=diag_float["spinodal_phi_at_min_fpp"],
        diag_fpp_min_abs=diag_float["fpp_min_abs"],
        diag_phi_at_min_abs_fpp=diag_float["phi_at_min_abs_fpp"],
        diag_success_fraction=diag_float["success_fraction"],
        diag_max_residual=diag_float["max_residual"],
        diag_mean_residual=diag_float["mean_residual"],
        diag_median_residual=diag_float["median_residual"],
        diag_phi_at_max_residual=diag_float["phi_at_max_residual"],
        diag_barrier=diag_float["barrier"], diag_phi1=diag_float["phi1"], diag_phi2=diag_float["phi2"],
        diag_delta_min=diag_float["delta_min"], diag_delta_max=diag_float["delta_max"],
        diag_A_min=diag_float["A_min"], diag_A_max=diag_float["A_max"],
        diag_min_f=diag_float["min_f"], diag_max_f=diag_float["max_f"],
        diag_min_mu=diag_float["min_mu"], diag_max_mu=diag_float["max_mu"],
        diag_e_linear_min=diag_float["e_linear_min"],
        diag_e_linear_max=diag_float["e_linear_max"],
        diag_avg_Eeff_min=diag_float["avg_Eeff_min"],
        diag_avg_Eeff_max=diag_float["avg_Eeff_max"],
        diag_class_free_energy_min=diag_float["class_free_energy_min"],
        diag_class_free_energy_max=diag_float["class_free_energy_max"],
        diag_logZ_internal_min=diag_float["logZ_internal_min"],
        diag_logZ_internal_max=diag_float["logZ_internal_max"],
        diag_bethe_raw_median=diag_float["bethe_raw_median"],
        diag_bethe_corr_median=diag_float["bethe_corr_median"],
        diag_bethe_allowed_prob_median=diag_float["bethe_allowed_prob_median"],
        diag_bethe_ok_fraction=diag_float["bethe_ok_fraction"],
        shard_id=np.array([args.shard_id], dtype=np.int64),

        start=np.array([start], dtype=np.int64),
        stop=np.array([stop], dtype=np.int64),
    )

    meta = {
        "shard_id": int(args.shard_id),
        "n_shards": int(args.n_shards),
        "start": int(start), "stop": int(stop),
        "processed": int(stop - start),
        "n_found": int(len(found)),
        "n_evaluated": int(len(all_results)),
        "factor": float(args.factor),
        "linear_term_mode": str(args.linear_term_mode),
        "registry_mode": str(args.registry_mode),
        "association_mode": str(args.association_mode),
        "bethe_correction": str(args.bethe_correction),
        "bethe_strength": float(args.bethe_strength),
        "bethe_contact_factor": float(args.bethe_contact_factor),
        "bethe_threshold": float(args.bethe_threshold),
        "bethe_compat_mode": str(args.bethe_compat_mode),
        "bethe_orientation_mode": str(args.bethe_orientation_mode),
        "bethe_max_iter": int(args.bethe_max_iter),
        "bethe_tol": float(args.bethe_tol),
        "bethe_sc_max_iter": int(args.bethe_sc_max_iter),
        "bethe_sc_tol": float(args.bethe_sc_tol),
        "bethe_sc_damping": float(args.bethe_sc_damping),
        "use_boundary_quotient": bool(args.use_boundary_quotient),
        "n_spinodal": int(sum(1 for d in found if d.get("spinodal_detected", False))),
        "n_binodal": int(sum(1 for d in found if d.get("binodal_detected", False))),
        "elapsed_sec": float(time.perf_counter() - t0),
        "out_npz": str(out_npz),
    }
    out_json.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    log(f"shard {args.shard_id}: saved {out_npz} with {len(found)} points")


def cmd_merge_results(args):
    out_dir = Path(args.out_dir)
    shard_files = sorted(out_dir.glob("shard_*.npz"))
    if not shard_files:
        raise FileNotFoundError(f"no shard_*.npz in {out_dir}")

    all_points, all_phi1, all_phi2 = [], [], []
    all_barrier, all_eps_a, all_eps_c = [], [], []
    all_binodal_detected, all_spinodal_detected = [], []
    all_spinodal_n_crossings, all_spinodal_min_fpp, all_spinodal_phi_at_min_fpp = [], [], []
    all_spinodal_phis = []
    diag_names = [
        "diag_eps_a", "diag_eps_c", "diag_event_detected", "diag_spinodal_detected",
        "diag_binodal_detected", "diag_n_phi", "diag_n_valid_phi", "diag_n_success_phi",
        "diag_n_phi_requested", "diag_n_phi_invalid",
        "diag_status", "diag_linear_term_mode", "diag_registry_mode", "diag_bethe_correction",
        "diag_bethe_compat_mode", "diag_bethe_orientation_mode", "diag_bethe_gamma_status",
        "diag_spinodal_min_fpp", "diag_spinodal_phi_at_min_fpp",
        "diag_fpp_min_abs", "diag_phi_at_min_abs_fpp", "diag_success_fraction",
        "diag_max_residual", "diag_mean_residual", "diag_median_residual",
        "diag_phi_at_max_residual", "diag_barrier", "diag_phi1", "diag_phi2",
        "diag_delta_min", "diag_delta_max", "diag_A_min", "diag_A_max",
        "diag_min_f", "diag_max_f", "diag_min_mu", "diag_max_mu",
        "diag_e_linear_min", "diag_e_linear_max",
        "diag_avg_Eeff_min", "diag_avg_Eeff_max",
        "diag_class_free_energy_min", "diag_class_free_energy_max",
        "diag_logZ_internal_min", "diag_logZ_internal_max",
        "diag_bethe_raw_median", "diag_bethe_corr_median",
        "diag_bethe_allowed_prob_median", "diag_bethe_ok_fraction",
    ]
    all_diag = {name: [] for name in diag_names}

    for fp in shard_files:
        z = np.load(fp)
        if "points" in z and z["points"].size:
            all_points.append(np.asarray(z["points"], dtype=np.float64))
            all_phi1.append(np.asarray(z["binodal_phi1"], dtype=np.float64))
            all_phi2.append(np.asarray(z["binodal_phi2"], dtype=np.float64))
            all_barrier.append(np.asarray(z["binodal_barrier"], dtype=np.float64))
            all_eps_a.append(np.asarray(z["binodal_eps_a"], dtype=np.float64))
            all_eps_c.append(np.asarray(z["binodal_eps_c"], dtype=np.float64))
            all_binodal_detected.append(np.asarray(z["binodal_detected"], dtype=bool))
            all_spinodal_detected.append(np.asarray(z["spinodal_detected"], dtype=bool))
            all_spinodal_n_crossings.append(np.asarray(z["spinodal_n_crossings"], dtype=np.int64))
            all_spinodal_min_fpp.append(np.asarray(z["spinodal_min_fpp"], dtype=np.float64))
            all_spinodal_phi_at_min_fpp.append(np.asarray(z["spinodal_phi_at_min_fpp"], dtype=np.float64))
            all_spinodal_phis.append(np.asarray(z["spinodal_phis"], dtype=np.float64))
        for name in diag_names:
            if name in z:
                all_diag[name].append(np.asarray(z[name]))

    if all_points:
        points = np.vstack(all_points)
        binodal_phi1 = np.concatenate(all_phi1)
        binodal_phi2 = np.concatenate(all_phi2)
        binodal_barrier = np.concatenate(all_barrier)
        binodal_eps_a = np.concatenate(all_eps_a)
        binodal_eps_c = np.concatenate(all_eps_c)
        binodal_detected = np.concatenate(all_binodal_detected)
        spinodal_detected = np.concatenate(all_spinodal_detected)
        spinodal_n_crossings = np.concatenate(all_spinodal_n_crossings)
        spinodal_min_fpp = np.concatenate(all_spinodal_min_fpp)
        spinodal_phi_at_min_fpp = np.concatenate(all_spinodal_phi_at_min_fpp)
        max_cols = max([a.shape[1] if a.ndim == 2 else 0 for a in all_spinodal_phis] + [0])
        padded = []
        for a in all_spinodal_phis:
            b = np.full((a.shape[0], max_cols), np.nan, dtype=np.float64)
            if a.ndim == 2 and a.shape[1] > 0:
                b[:, :a.shape[1]] = a
            padded.append(b)
        spinodal_phis = np.vstack(padded) if padded else np.empty((0, 0), dtype=np.float64)
        order = np.lexsort((points[:, 0], points[:, 1]))
        points = points[order]
        binodal_phi1 = binodal_phi1[order]
        binodal_phi2 = binodal_phi2[order]
        binodal_barrier = binodal_barrier[order]
        binodal_eps_a = binodal_eps_a[order]
        binodal_eps_c = binodal_eps_c[order]
        binodal_detected = binodal_detected[order]
        spinodal_detected = spinodal_detected[order]
        spinodal_n_crossings = spinodal_n_crossings[order]
        spinodal_min_fpp = spinodal_min_fpp[order]
        spinodal_phi_at_min_fpp = spinodal_phi_at_min_fpp[order]
        spinodal_phis = spinodal_phis[order]
    else:
        points = np.empty((0, 2), dtype=np.float64)
        binodal_phi1 = np.array([], dtype=np.float64)
        binodal_phi2 = np.array([], dtype=np.float64)
        binodal_barrier = np.array([], dtype=np.float64)
        binodal_eps_a = np.array([], dtype=np.float64)
        binodal_eps_c = np.array([], dtype=np.float64)
        binodal_detected = np.array([], dtype=bool)
        spinodal_detected = np.array([], dtype=bool)
        spinodal_n_crossings = np.array([], dtype=np.int64)
        spinodal_min_fpp = np.array([], dtype=np.float64)
        spinodal_phi_at_min_fpp = np.array([], dtype=np.float64)
        spinodal_phis = np.empty((0, 0), dtype=np.float64)

    merged_diag = {}
    for name in diag_names:
        if all_diag[name]:
            merged_diag[name] = np.concatenate(all_diag[name])
        else:
            if name in ("diag_status", "diag_linear_term_mode", "diag_registry_mode", "diag_bethe_correction", "diag_bethe_compat_mode", "diag_bethe_orientation_mode", "diag_bethe_gamma_status"):
                merged_diag[name] = np.array([], dtype=str)
            elif name.startswith("diag_n_"):
                merged_diag[name] = np.array([], dtype=np.int64)
            elif name.endswith("detected"):
                merged_diag[name] = np.array([], dtype=bool)
            else:
                merged_diag[name] = np.array([], dtype=np.float64)

    if merged_diag["diag_eps_a"].size:
        diag_order = np.lexsort((merged_diag["diag_eps_a"], merged_diag["diag_eps_c"]))
        for name in diag_names:
            merged_diag[name] = merged_diag[name][diag_order]

    eps_as, eps_cs = grid_from_args(args)
    final_npz = Path(args.final_npz)
    final_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        final_npz,
        points=points, eps_as=eps_as, eps_cs=eps_cs,
        binodal_phi1=binodal_phi1, binodal_phi2=binodal_phi2,
        binodal_barrier=binodal_barrier,
        binodal_eps_a=binodal_eps_a, binodal_eps_c=binodal_eps_c,
        binodal_detected=binodal_detected,
        spinodal_detected=spinodal_detected,
        spinodal_n_crossings=spinodal_n_crossings,
        spinodal_min_fpp=spinodal_min_fpp,
        spinodal_phi_at_min_fpp=spinodal_phi_at_min_fpp,
        spinodal_phis=spinodal_phis,
        **merged_diag,
    )

    meta = {
        "n_shards_found": len(shard_files),
        "n_points": int(len(points)),
        "n_spinodal": int(np.count_nonzero(spinodal_detected)),
        "n_binodals": int(np.count_nonzero(binodal_detected)),
        "eps_as_range": [float(eps_as.min()), float(eps_as.max())],
        "eps_cs_range": [float(eps_cs.min()), float(eps_cs.max())],
        "n_eps_a": int(len(eps_as)),
        "n_eps_c": int(len(eps_cs)),
        "n_phis": int(args.n_phis),
        "factor": float(args.factor),
        "linear_term_mode": str(getattr(args, "linear_term_mode", "logZnu")),
        "registry_mode": str(getattr(args, "registry_mode", "boltzmann")),
        "use_boundary_quotient": bool(getattr(args, "use_boundary_quotient", False)),
        "bethe_correction": str(getattr(args, "bethe_correction", "none")),
        "bethe_strength": float(getattr(args, "bethe_strength", 0.0)),
        "bethe_contact_factor": float(getattr(args, "bethe_contact_factor", 1.0)),
        "bethe_threshold": float(getattr(args, "bethe_threshold", 1e-12)),
        "bethe_compat_mode": str(getattr(args, "bethe_compat_mode", "slot_exact")),
        "bethe_orientation_mode": str(getattr(args, "bethe_orientation_mode", "orbit_average")),
        "bethe_max_iter": int(getattr(args, "bethe_max_iter", 500)),
        "bethe_tol": float(getattr(args, "bethe_tol", 1e-10)),
        "bethe_sc_max_iter": int(getattr(args, "bethe_sc_max_iter", 20)),
        "bethe_sc_tol": float(getattr(args, "bethe_sc_tol", 1e-5)),
        "bethe_sc_damping": float(getattr(args, "bethe_sc_damping", 0.5)),
        "n_diagnostics": int(merged_diag.get("diag_eps_a", np.array([])).size),
    }
    meta_path = final_npz.with_suffix(final_npz.suffix + ".json")
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print("merged shard files =", len(shard_files))
    print("points.shape =", points.shape)
    print("n_spinodal =", int(np.count_nonzero(spinodal_detected)))
    print("n_binodal =", int(np.count_nonzero(binodal_detected)))
    print("n_diagnostics =", int(merged_diag.get("diag_eps_a", np.array([])).size))
    print("saved =", str(final_npz))


def add_grid_args(p):
    p.add_argument("--eps-a-min", type=float, required=True)
    p.add_argument("--eps-a-max", type=float, required=True)
    p.add_argument("--n-eps-a", type=int, required=True)
    p.add_argument("--eps-c-min", type=float, required=True)
    p.add_argument("--eps-c-max", type=float, required=True)
    p.add_argument("--n-eps-c", type=int, required=True)
    p.add_argument("--phi-min", type=float, default=0.01)
    p.add_argument("--phi-max", type=float, default=0.99)
    p.add_argument("--n-phis", type=int, default=31)
    p.add_argument("--factor", type=float, default=6.0)
    p.add_argument("--linear-term-mode", choices=("logZnu", "logZ", "avgE"), default="logZnu",
                   help="Internal class treatment: logZnu puts the composition multiplier nu inside each class partition function (recommended); logZ and avgE retain the legacy fixed-class approximations.")
    p.add_argument("--registry-mode", choices=("boltzmann", "min"), default="boltzmann",
                   help="Face-face registry closure: boltzmann uses -log(mean exp[-E_r]); min uses the most-compatible/minimum-energy registry.")
    p.add_argument("--association-mode", choices=("global_face", "directed_face_state", "oriented_directed_face_state", "explicit_oriented_directed_face_state"), default="oriented_directed_face_state",
                   help="Association closure. global_face is original SAFT-P X_s. directed_face_state pins canonical representatives to lab directions. oriented_directed_face_state uses the same oriented (g,R) theory as before but evaluates the 24 orientations implicitly to avoid the 24x memory blowup. explicit_oriented_directed_face_state is the old memory-heavy reference path.")
    p.add_argument("--use-boundary-quotient", dest="use_boundary_quotient", action="store_true", default=False)
    p.add_argument("--no-boundary-quotient", dest="use_boundary_quotient", action="store_false")
    p.add_argument("--bethe-correction", choices=("none", "boundary_entropy", "boundary_bethe", "face_entropy", "face_bethe", "selfconsistent_boundary_bethe", "sc_boundary_bethe", "cavity_bethe"), default="none",
                   help="Compatibility correction. selfconsistent_boundary_bethe inserts the boundary Bethe cavity field inside the rho minimization; boundary_bethe/face_bethe are older post-solve diagnostics.")
    p.add_argument("--bethe-strength", type=float, default=0.0,
                   help="Only used by post-solve Bethe modes. Ignored by selfconsistent_boundary_bethe, whose coefficient is fixed by the 1/8 cube-size factor.")
    p.add_argument("--bethe-contact-factor", type=float, default=1.0,
                   help="Extra multiplicative contact factor for the correction before division by 8.")
    p.add_argument("--bethe-threshold", type=float, default=1e-12,
                   help="Only used for --bethe-compat-mode attractive: exp(-eps_face)-1 > threshold.")
    p.add_argument("--bethe-compat-mode", choices=("slot_exact", "slot_presence", "slot_nonconflict", "attractive", "all"), default="attractive",
                   help="Hard support rule. Default attractive makes the correction vanish at eps_a=eps_c=0; slot_exact is a strict diagnostic only and can create unphysical ideal-state curvature.")
    p.add_argument("--bethe-orientation-mode", choices=("orbit_average", "representative"), default="orbit_average",
                   help="How to handle canonical representative orientations. orbit_average restores rotational invariance; representative uses the stored face order directly.")
    p.add_argument("--bethe-max-iter", type=int, default=500,
                   help="Max Sinkhorn/IPF iterations for Bethe compatibility corrections.")
    p.add_argument("--bethe-tol", type=float, default=1e-10,
                   help="Sinkhorn/IPF marginal tolerance for Bethe compatibility corrections.")
    p.add_argument("--bethe-sc-max-iter", type=int, default=20,
                   help="Maximum outer fixed-point iterations for selfconsistent_boundary_bethe.")
    p.add_argument("--bethe-sc-tol", type=float, default=1e-5,
                   help="Convergence tolerance on the self-consistent Bethe cavity field pi_g.")
    p.add_argument("--bethe-sc-damping", type=float, default=0.5,
                   help="Damping for the self-consistent Bethe cavity-field update.")


def build_parser():
    ap = argparse.ArgumentParser(description="Cube spinodal scan: representative SAFT-P + optional boundary-state Bethe compatibility correction.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("prepare-mmap")
    p.add_argument("--cache-path", required=True)
    p.add_argument("--mmap-dir", required=True)

    p = sub.add_parser("run-shard")
    p.add_argument("--patches", required=True)
    p.add_argument("--cache-path", required=True)
    p.add_argument("--mmap-dir", default=None)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--shard-id", type=int, required=True)
    p.add_argument("--n-shards", type=int, required=True)
    p.add_argument("--tol-solver", type=float, default=1e-8)
    p.add_argument("--accept-residual", type=float, default=1e-4)
    p.add_argument("--max-iter-solver", type=int, default=30)
    p.add_argument("--max-jac-rebuilds", type=int, default=1)
    p.add_argument("--fallback-newton", action="store_true")
    add_grid_args(p)

    p = sub.add_parser("merge-results")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--final-npz", required=True)
    add_grid_args(p)

    return ap


def main():
    args = build_parser().parse_args()
    if args.cmd == "prepare-mmap":
        cmd_prepare_mmap(args)
    elif args.cmd == "run-shard":
        cmd_run_shard(args)
    elif args.cmd == "merge-results":
        cmd_merge_results(args)
    else:
        raise SystemExit(f"Unknown command: {args.cmd}")


if __name__ == "__main__":
    main()
