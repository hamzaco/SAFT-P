"""
Thin driver: build n x n plaquette classes for a (patches, rot90_species)
geometry at given (eps_a, eps_c), solve the reduced SAFT-P free-energy curve,
and extract the spinodal / binodal exactly as the published notebooks do.
"""

from __future__ import annotations

import time

import numpy as np

from plaquette_nxn_by_species import build_plaquettes_nxn_by_species
from saftp_reduced_nxn import (
    build_S_matrix,
    solve_free_energy_curve,
    loess_derivs,
    zero_crossings_linear,
    extract_binodals_from_convex_envelope,
    best_binodal_segment,
)

# Geometries used in the manuscript.  patches rows are (N, E, S, W) patch ids;
# the last species is the vacancy.  Patch id 1 is the sticky patch, 0 the inert
# side of the particle, 2 the vacancy patch.
GEOMETRIES = {
    "stick": dict(
        patches=np.array([[1, 0, 1, 0],
                          [0, 1, 0, 1],
                          [2, 2, 2, 2]], dtype=np.int64),
        rot90_species=np.array([1, 0, 2], dtype=np.int64),
    ),
    "L": dict(
        patches=np.array([[1, 1, 0, 0],
                          [0, 1, 1, 0],
                          [0, 0, 1, 1],
                          [1, 0, 0, 1],
                          [2, 2, 2, 2]], dtype=np.int64),
        rot90_species=np.array([1, 2, 3, 0, 4], dtype=np.int64),
    ),
}


def make_J(patches: np.ndarray, eps_a: float, eps_c: float) -> np.ndarray:
    """Same J as the published notebooks."""
    M = int(np.max(patches)) + 1
    J = np.zeros((M, M), dtype=np.float64)
    J[1, 1] = -float(eps_a)
    J[:-1, :-1] -= float(eps_c)
    return J


def build_reduced_inputs(patches, rot90_species, eps_a, eps_c, *, n=3, factor=4.0,
                         interior_key="occupancy", internal_term="mean_energy",
                         composition_key="components"):
    """
    Returns (A_row, e_linear, log_mult, S, delta, nsite, info).

    ``factor`` is the number of boundary super-edges per plaquette (4 for any
    square plaquette), so it does NOT change between 2x2 and 3x3.

    ``composition_key="components"`` (the default) makes the plaquette classes
    composition-aware, so ``A_row`` is an exact multiple of 1/n^2 rather than a
    Boltzmann average over microstates with different particle counts.  Pass
    ``composition_key="none"`` to reproduce the published class definition.
    ``info["A_row_max_noninteger"]`` reports how far the class compositions are
    from integers and must be 0 for the default.
    """
    J = make_J(patches, eps_a, eps_c)
    t0 = time.time()
    (eps_small, intra_bonds, plaq_to_species, m_patch,
     patch_to_species, patch_to_small, plaq_configs, mult) = build_plaquettes_nxn_by_species(
        patches, J, np.zeros(len(patches)), rot90_species=rot90_species, n=n,
        interior_key=interior_key, internal_term=internal_term,
        composition_key=composition_key)
    t_build = time.time() - t0

    nsite = n * n
    A_row = np.sum(plaq_to_species[:, :-1], axis=1) / float(nsite)
    P = int(len(A_row))
    Fd = int(np.max(patch_to_small)) + 1
    S = build_S_matrix(patch_to_species, patch_to_small, m_patch, P, Fd)
    log_mult = np.log(mult.astype(np.float64) + 1e-300)
    delta = (np.exp(-eps_small) - 1.0) / float(factor)

    n_occ = A_row * float(nsite)
    info = {"P": P, "F": Fd, "build_time": t_build, "n": n, "nsite": nsite,
            "interior_key": interior_key, "internal_term": internal_term,
            "composition_key": composition_key,
            "phi_min": float(A_row.min()),
            "phi_max": float(A_row.max()),
            # 0 exactly when every class holds an integer number of particles.
            "A_row_max_noninteger": float(np.abs(n_occ - np.rint(n_occ)).max()),
            "plaq_configs": plaq_configs, "eps_small": eps_small}
    return A_row, intra_bonds.astype(np.float64), log_mult, S, delta, nsite, info


def free_energy_curve(geometry, eps_a, eps_c, *, n=3, phis=None, factor=4.0,
                      tol=1e-8, max_newton=40, logW=True, mu_W_init=None,
                      inputs=None, interior_key="occupancy",
                      internal_term="mean_energy", composition_key="components",
                      parallel=False):
    g = GEOMETRIES[geometry] if isinstance(geometry, str) else geometry
    if inputs is None:
        inputs = build_reduced_inputs(g["patches"], g["rot90_species"],
                                      eps_a, eps_c, n=n, factor=factor,
                                      interior_key=interior_key,
                                      internal_term=internal_term,
                                      composition_key=composition_key)
    A_row, e_lin, log_mult, S, delta, nsite, info = inputs
    phis = np.linspace(0.01, 0.99, 151) if phis is None else np.asarray(phis, float)

    mu0, W0 = (0.0, None) if mu_W_init is None else mu_W_init
    t0 = time.time()
    curve = solve_free_energy_curve(phis, A_row, e_lin, log_mult, S, delta, nsite,
                                    mu_init=mu0, W_init=W0, max_newton=max_newton,
                                    tol=tol, accept_residual=max(1e-6, 10.0 * tol),
                                    logW=logW, parallel=parallel)
    curve["solve_time"] = time.time() - t0
    curve["info"] = info
    curve["eps_a"] = float(eps_a)
    curve["eps_c"] = float(eps_c)
    return curve


def analyse_curve(curve, *, margin_frac=4, coexist_tol=1e-5, min_gap_points=6,
                  bandwidth=0.15):
    """Spinodal roots and the best common-tangent segment, as in the notebooks."""
    phi = np.asarray(curve["phi"], float)
    f = np.asarray(curve["f"], float)
    valid = ~np.isnan(f)
    phi_v, f_v = phi[valid], f[valid]
    out = {"phi": phi_v, "f": f_v, "spinodal_phis": [], "binodal": None,
           "n_valid": int(valid.sum())}
    if len(phi_v) < 5:
        return out

    margin = min(2, len(phi_v) // margin_frac)
    phi_in = phi_v[margin:-margin] if margin > 0 else phi_v
    f_in = f_v[margin:-margin] if margin > 0 else f_v
    if len(phi_in) < 5:
        return out

    f_s, _, fpp = loess_derivs(phi_in, f_in, bandwidth=bandwidth)
    out["phi_inner"] = phi_in
    out["f_smooth"] = f_s
    out["fpp"] = fpp
    out["spinodal_phis"] = zero_crossings_linear(phi_in, -fpp, tol=1e-6)

    res = extract_binodals_from_convex_envelope(phi_in, f_s, coexist_tol=coexist_tol,
                                                min_gap_points=min_gap_points)
    out["binodal"] = best_binodal_segment(res, prefer="largest_barrier")
    return out


def has_two_phase(analysis, *, barrier_tol=1e-4, require_two_spinodal=False):
    b = analysis.get("binodal")
    if b is None or b["barrier"] <= barrier_tol:
        return False
    if require_two_spinodal and len(analysis.get("spinodal_phis", [])) <= 1:
        return False
    return True


def bisect_critical_eps_a(geometry, eps_c, *, n=3, lo=0.0, hi=10.0, iters=18,
                          phis=None, factor=4.0, tol=1e-8, max_newton=40,
                          barrier_tol=1e-4, require_two_spinodal=False,
                          interior_key="occupancy",
                          internal_term="mean_energy",
                          composition_key="components", parallel=False,
                          verbose=False):
    """
    Smallest eps_a at fixed eps_c for which the free-energy curve develops a
    common-tangent construction (the critical line in the (eps_c, eps_a) plane).

    Assumes monotonicity in eps_a, which is what the published scans show.
    """
    g = GEOMETRIES[geometry] if isinstance(geometry, str) else geometry

    def two_phase(ea):
        c = free_energy_curve(g, ea, eps_c, n=n, phis=phis, factor=factor,
                              tol=tol, max_newton=max_newton,
                              interior_key=interior_key,
                              internal_term=internal_term,
                              composition_key=composition_key, parallel=parallel)
        a = analyse_curve(c)
        ok = has_two_phase(a, barrier_tol=barrier_tol,
                           require_two_spinodal=require_two_spinodal)
        if verbose:
            b = a.get("binodal")
            print(f"    eps_a={ea:7.4f}  two_phase={ok}  "
                  f"barrier={(b['barrier'] if b else 0.0):.3e}  "
                  f"nspin={len(a['spinodal_phis'])}  ({c['solve_time']:.1f}s)")
        return ok

    if two_phase(lo):
        return float(lo)
    if not two_phase(hi):
        return float("nan")
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if two_phase(mid):
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def map_two_phase_region(geometry, eps_cs, eps_as, *, n=3, phis=None, factor=4.0,
                         tol=1e-8, max_newton=40, interior_key="occupancy",
                         internal_term="mean_energy", composition_key="components",
                         parallel=False, verbose=True):
    """
    Sweep a (eps_c, eps_a) grid and record, for each point, the best common-tangent
    barrier and the number of spinodal roots.

    Use this rather than ``bisect_critical_eps_a`` whenever the two-phase region
    may not be a half-plane in eps_a.  It is for the L-shaped particle: its
    two-phase region is a *band*, so at fixed eps_c there is both a lower and an
    upper critical eps_a and a bisection that assumes monotonicity is invalid.
    The stick region is a half-plane in eps_a and either routine works there.

    Returns dict with 2-D arrays indexed [i_eps_c, j_eps_a]:
        barrier, n_spinodal, phi1, phi2, n_valid, solve_time
    """
    g = GEOMETRIES[geometry] if isinstance(geometry, str) else geometry
    eps_cs = np.asarray(eps_cs, float)
    eps_as = np.asarray(eps_as, float)
    shp = (len(eps_cs), len(eps_as))
    out = {k: np.full(shp, np.nan) for k in ("barrier", "phi1", "phi2")}
    out["n_spinodal"] = np.zeros(shp, dtype=np.int64)
    out["n_valid"] = np.zeros(shp, dtype=np.int64)
    out["solve_time"] = np.zeros(shp)

    for i, ec in enumerate(eps_cs):
        for j, ea in enumerate(eps_as):
            c = free_energy_curve(g, ea, ec, n=n, phis=phis, factor=factor, tol=tol,
                                  max_newton=max_newton, interior_key=interior_key,
                                  internal_term=internal_term,
                                  composition_key=composition_key, parallel=parallel)
            a = analyse_curve(c)
            b = a["binodal"]
            out["barrier"][i, j] = b["barrier"] if b else 0.0
            out["phi1"][i, j] = b["phi1"] if b else np.nan
            out["phi2"][i, j] = b["phi2"] if b else np.nan
            out["n_spinodal"][i, j] = len(a["spinodal_phis"])
            out["n_valid"][i, j] = a["n_valid"]
            out["solve_time"][i, j] = c["solve_time"]
            if verbose:
                print(f"  eps_c={ec:5.2f} eps_a={ea:5.2f}  barrier={out['barrier'][i,j]:.2e}  "
                      f"nspin={out['n_spinodal'][i,j]}  nvalid={out['n_valid'][i,j]:3d}  "
                      f"({c['solve_time']:5.1f}s)", flush=True)
    out["eps_cs"] = eps_cs
    out["eps_as"] = eps_as
    return out


def two_phase_mask(region, *, barrier_tol=1e-4, require_two_spinodal=False):
    m = region["barrier"] > barrier_tol
    if require_two_spinodal:
        m &= region["n_spinodal"] > 1
    return m


def critical_eps_a_lower_edge(geometry, eps_c, *, n=3, eps_a_scan=None, refine=6,
                              phis=None, factor=4.0, tol=1e-8, max_newton=40,
                              interior_key="occupancy", internal_term="mean_energy",
                              composition_key="components",
                              barrier_tol=1e-4, require_two_spinodal=False,
                              parallel=False, verbose=False):
    """
    Lower edge of the two-phase region in eps_a at fixed eps_c, found by a coarse
    scan followed by bisection between the last one-phase point and the first
    two-phase point.

    Unlike ``bisect_critical_eps_a`` this does NOT assume the region is a
    half-plane in eps_a, so it is safe for the L-shaped particle whose two-phase
    region is a band.  It only assumes the *lower* edge is a single crossing,
    which is what the published scans read off (min eps_a at each eps_c).

    Returns (eps_a_lower, info dict).
    """
    g = GEOMETRIES[geometry] if isinstance(geometry, str) else geometry
    if eps_a_scan is None:
        eps_a_scan = np.array([0., 0.5, 1., 1.5, 2., 2.5, 3., 4., 5., 6., 7., 8., 9., 10.])
    eps_a_scan = np.asarray(eps_a_scan, float)

    def two_phase(ea):
        c = free_energy_curve(g, ea, eps_c, n=n, phis=phis, factor=factor, tol=tol,
                              max_newton=max_newton, interior_key=interior_key,
                              internal_term=internal_term,
                              composition_key=composition_key, parallel=parallel)
        a = analyse_curve(c)
        ok = has_two_phase(a, barrier_tol=barrier_tol,
                           require_two_spinodal=require_two_spinodal)
        b = a["binodal"]
        if verbose:
            print(f"    eps_a={ea:6.3f} two_phase={ok!s:5s} "
                  f"barrier={(b['barrier'] if b else 0.0):.2e} "
                  f"nspin={len(a['spinodal_phis'])} ({c['solve_time']:.1f}s)", flush=True)
        return ok, (b["barrier"] if b else 0.0)

    lo = None
    hi = None
    scanned = []
    for ea in eps_a_scan:
        ok, bar = two_phase(ea)
        scanned.append((float(ea), bool(ok), float(bar)))
        if ok:
            hi = float(ea)
            break
        lo = float(ea)

    info = {"scan": scanned, "eps_c": float(eps_c), "n": n,
            "interior_key": interior_key, "internal_term": internal_term,
            "composition_key": composition_key}
    if hi is None:
        info["status"] = "no two-phase point in scan"
        return float("nan"), info
    if lo is None:
        info["status"] = "two-phase already at the lowest eps_a scanned"
        return float(hi), info

    for _ in range(refine):
        mid = 0.5 * (lo + hi)
        ok, _ = two_phase(mid)
        if ok:
            hi = mid
        else:
            lo = mid
    info["status"] = "ok"
    info["bracket"] = [lo, hi]
    return 0.5 * (lo + hi), info
