"""
Mercedes (3-patch) particle on a triangular lattice, 7-site compact hexagon.

Wraps ``plaquette_by_species_hexagon`` so the same three questions asked of the
square 2x2 / 3x3 plaquettes can be asked here:

*  what happens to the centre site, which the boundary-face signature cannot see;
*  what goes in the class exponent;
*  where does the resulting critical line sit relative to Monte Carlo.

The 7-site hexagon is 1 centre + 6 ring sites, 12 internal bonds (6 centre-ring,
6 ring-ring) and 6 boundary faces, each face an ordered 3-slot super-patch, so
``factor = 6`` and ``nsite = 7``.

Centre treatment
    ``split_center_vacancy=False``  (the published setting) Boltzmann-averages the
    centre.  Its occupancy is then pinned near the ideal value and the reachable
    composition window collapses to phi in [2/21, 1 - 1/21] = [0.0952, 0.9524] at
    zero coupling -- the theory has neither a pure-solvent nor a pure-condensate
    state.
    ``split_center_vacancy=True`` splits each boundary class by whether the centre
    holds the vacancy, restoring phi in [0, 1] at exactly twice the class count.

Class exponent
    ``hexagon7_from_cache`` sets ``intra_bonds = min_e``, the class *ground-state*
    energy, with multiplicity 1.  Note this is a third convention: the square
    builder uses the Boltzmann-averaged energy <E>_p, and the thermodynamically
    consistent choice is g_p = -log Z_p.  The cache already computes the weight
    sum w needed for the latter, so all three are available here at no extra cost:

        min_energy         e_p = E0
        mean_energy        e_p = sum_i w_i E_i / sum_i w_i
        class_free_energy  e_p = E0 - log(sum_i exp(-(E_i - E0)))  =  -log Z_p
"""

from __future__ import annotations

import time
from typing import Dict, Any

import numpy as np

from plaquette_by_species_hexagon import (
    build_hexagon7_geometry_cache,
    hexagon7_from_cache,
)
# Acceptance protocol, identical to spinodal_stick_shaped / spinodal_l_shaped and
# to the scan cell of spinodal_hex_mercedes.ipynb: a segment survives extraction if
# its barrier exceeds COEXIST_TOL, the largest-barrier survivor is selected, and the
# state point is accepted if that barrier exceeds ACCEPT_BARRIER.  Effective
# threshold = max of the two = 1e-4.  These defaults used to be coexist_tol=1e-3 and
# barrier_tol=1e-4, i.e. an effective 1e-3 -- ten times looser than the scan, so this
# module and the scan disagreed about where the critical line was.
COEXIST_TOL = 1e-5
ACCEPT_BARRIER = 1e-4
REQUIRE_TWO_SPINODAL = False   # the square-lattice scans do not impose this

from saftp_reduced_nxn import (
    build_S_matrix,
    solve_free_energy_curve,
    loess_derivs,
    zero_crossings_linear,
    extract_binodals_from_convex_envelope,
    best_binodal_segment,
)

# Mercedes: three sticky patches at 120 deg, two orientations plus a vacancy.
PATCHES = np.array([[1, 0, 1, 0, 1, 0],
                    [0, 1, 0, 1, 0, 1],
                    [2, 2, 2, 2, 2, 2]], dtype=np.int64)
ROT60 = np.array([1, 0, 2], dtype=np.int64)
NSITE = 7
FACTOR = 6.0

_CACHE: Dict[Any, dict] = {}


def get_cache(split_center_vacancy: bool, composition_key: str = "components") -> dict:
    """
    Geometry cache for the 7-site hexagon.

    ``composition_key="components"`` (the default) keeps hexagons with different
    particle counts in different classes, so the class composition is an exact
    multiple of 1/7 and the reachable phi spans [0, 1].  With the legacy
    ``"none"`` key the boundary faces are built from ring sites only, the centre
    is invisible, and the classes carry Boltzmann-averaged fractional
    compositions that never reach either pure phase.  Pass ``"none"`` to
    reproduce the published construction.
    """
    ck = (bool(split_center_vacancy), str(composition_key))
    if ck not in _CACHE:
        _CACHE[ck] = build_hexagon7_geometry_cache(
            PATCHES, rot60_species=ROT60,
            canonicalize_by_boundary_edges=True,
            split_center_vacancy=split_center_vacancy,
            composition_key=composition_key)
    return _CACHE[ck]


def make_J(eps_a: float, eps_c: float) -> np.ndarray:
    M = int(PATCHES.max()) + 1
    J = np.zeros((M, M), dtype=np.float64)
    J[1, 1] = -float(eps_a)
    J[:-1, :-1] -= float(eps_c)
    return J


def class_terms(cache: dict, J: np.ndarray, internal_term: str = "min_energy"):
    """
    Recompute the per-class quantities for a new J, with a choice of what goes in
    the class exponent.  ``min_energy`` reproduces ``hexagon7_from_cache`` exactly.
    """
    bond_a, bond_b = cache["bond_a"], cache["bond_b"]
    group, n_groups = cache["group"], cache["n_groups"]
    Eeff = J[bond_a, bond_b].sum(axis=1)

    E0 = np.full(n_groups, np.inf)
    np.minimum.at(E0, group, Eeff)
    contrib = np.exp(-(Eeff - E0[group]))
    w = np.zeros(n_groups)
    np.add.at(w, group, contrib)
    Esum = np.zeros(n_groups)
    np.add.at(Esum, group, contrib * Eeff)

    if internal_term == "min_energy":
        e = E0.copy()
    elif internal_term == "mean_energy":
        e = Esum / w
    elif internal_term == "class_free_energy":
        e = E0 - np.log(w)
    else:
        raise ValueError("internal_term must be min_energy / mean_energy / class_free_energy")

    species_sum = np.zeros((n_groups, cache["Ssp"]))
    np.add.at(species_sum, group, cache["species_counts"] * contrib[:, None])
    hex_to_species = species_sum / w[:, None]

    fi0, fi1, fi2 = cache["fi0"], cache["fi1"], cache["fi2"]
    eps_small = (J[fi0[:, None], fi2[None, :]]
                 + J[fi1[:, None], fi1[None, :]]
                 + J[fi2[:, None], fi0[None, :]])
    return eps_small, e, hex_to_species, np.ones(n_groups)


def build_inputs(eps_a, eps_c, *, split_center_vacancy=False,
                 internal_term="min_energy", composition_key="components"):
    cache = get_cache(split_center_vacancy, composition_key)
    J = make_J(eps_a, eps_c)
    eps_small, e, hex_to_species, mult = class_terms(cache, J, internal_term)

    A_row = np.sum(hex_to_species[:, :-1], axis=1) / float(NSITE)
    P = cache["n_groups"]
    Fd = len(cache["unique_face_ids"])
    S = build_S_matrix(cache["patch_to_species"], cache["patch_to_small"],
                       cache["m_patch"], P, Fd)
    log_mult = np.log(mult + 1e-300)
    delta = (np.exp(-eps_small) - 1.0) / FACTOR
    n_occ = A_row * float(NSITE)
    info = {"P": P, "F": Fd, "phi_min": float(A_row.min()), "phi_max": float(A_row.max()),
            "split_center_vacancy": split_center_vacancy, "internal_term": internal_term,
            "composition_key": composition_key,
            # 0 exactly when every class holds an integer number of particles.
            "A_row_max_noninteger": float(np.abs(n_occ - np.rint(n_occ)).max())}
    return A_row, e, log_mult, S, delta, float(NSITE), info


def free_energy_curve(eps_a, eps_c, *, phis=None, split_center_vacancy=False,
                      internal_term="min_energy", composition_key="components",
                      tol=1e-8, max_newton=120, logW=False):
    # NOTE: logW=False (the direct-W Newton, which is also what the published
    # hexagon code uses) is the right default here.  Unlike the square case, some
    # face types have essentially zero abundance on the triangular lattice
    # (delta reaches ~3e2), the log-W residual then underflows, and 1-2 phi points
    # per curve fail to converge -- which makes the two-phase test flicker.
    phis = np.linspace(0.01, 0.99, 151) if phis is None else np.asarray(phis, float)
    inp = build_inputs(eps_a, eps_c, split_center_vacancy=split_center_vacancy,
                       internal_term=internal_term, composition_key=composition_key)
    t0 = time.time()
    curve = solve_free_energy_curve(phis, *inp[:6], max_newton=max_newton, tol=tol,
                                    accept_residual=max(1e-6, 10 * tol), logW=logW)
    curve["solve_time"] = time.time() - t0
    curve["info"] = inp[6]
    curve["eps_a"] = float(eps_a)
    curve["eps_c"] = float(eps_c)
    return curve


def analyse_curve(curve, *, coexist_tol=COEXIST_TOL, min_gap_points=6, bandwidth=0.15,
                  phi1_max=None):
    """
    ``phi1_max`` keeps only coexistence segments whose dilute branch sits below
    that composition, i.e. the gas-liquid transition.  On this system f(phi)
    carries a second, structurally different common tangent between two dense
    phases (phi1 ~ 0.45-0.94: the honeycomb-network vertex A|B.B.B. against the
    closed six-ring .|ABABAB), which the Monte-Carlo critical line does not
    track.  The two families are separated by a gap of 0.12 in phi1 -- three
    times the next-largest gap in the distribution, and never narrower than
    0.31 at any single eps_c -- so any cut in [0.33, 0.44] gives identical
    results; 0.40 is the midpoint.  ``None`` restores the unfiltered behaviour.
    """
    phi = np.asarray(curve["phi"], float)
    f = np.asarray(curve["f"], float)
    ok = ~np.isnan(f)
    phi_v, f_v = phi[ok], f[ok]
    out = {"spinodal_phis": [], "binodal": None, "n_valid": int(ok.sum())}
    if len(phi_v) < 5:
        return out
    margin = min(2, len(phi_v) // 4)
    phi_in = phi_v[margin:-margin] if margin > 0 else phi_v
    f_in = f_v[margin:-margin] if margin > 0 else f_v
    if len(phi_in) < 5:
        return out
    f_s, _, fpp = loess_derivs(phi_in, f_in, bandwidth=bandwidth)
    out.update(phi_inner=phi_in, f_smooth=f_s, fpp=fpp,
               spinodal_phis=zero_crossings_linear(phi_in, -fpp, tol=1e-6))
    res = extract_binodals_from_convex_envelope(phi_in, f_s, coexist_tol=coexist_tol,
                                                min_gap_points=min_gap_points)
    out["all_segments"] = list(res.get("segments", []))
    if phi1_max is not None:
        res["segments"] = [d for d in res["segments"] if d["phi1"] <= float(phi1_max)]
    out["n_rejected"] = len(out["all_segments"]) - len(res["segments"])
    out["binodal"] = best_binodal_segment(res, prefer="largest_barrier")
    return out


def critical_eps_a_lower_edge(eps_c, *, eps_a_scan=None, refine=8, phis=None,
                              split_center_vacancy=False, internal_term="min_energy",
                              composition_key="components",
                              barrier_tol=ACCEPT_BARRIER,
                              require_two_spinodal=REQUIRE_TWO_SPINODAL,
                              coexist_tol=COEXIST_TOL, phi1_max=None, tol=1e-8,
                              max_newton=120, logW=False, verbose=False):
    """Smallest eps_a with a common-tangent construction, by scan then bisection."""
    if eps_a_scan is None:
        eps_a_scan = np.array([0., 0.25, 0.5, 0.75, 1., 1.25, 1.5, 2., 2.5, 3., 3.5, 4., 5., 6.])
    eps_a_scan = np.asarray(eps_a_scan, float)

    def two_phase(ea):
        c = free_energy_curve(ea, eps_c, phis=phis,
                              split_center_vacancy=split_center_vacancy,
                              internal_term=internal_term,
                              composition_key=composition_key, tol=tol,
                              max_newton=max_newton, logW=logW)
        a = analyse_curve(c, coexist_tol=coexist_tol, phi1_max=phi1_max)
        b = a["binodal"]
        ok = b is not None and b["barrier"] > barrier_tol
        if ok and require_two_spinodal:
            ok = len(a["spinodal_phis"]) > 1
        if verbose:
            print(f"    eps_a={ea:6.3f} {ok!s:5s} barrier={(b['barrier'] if b else 0):.2e} "
                  f"nvalid={a['n_valid']:3d}", flush=True)
        return ok

    lo = hi = None
    scan = []
    for ea in eps_a_scan:
        ok = two_phase(ea)
        scan.append((float(ea), bool(ok)))
        if ok:
            hi = float(ea)
            break
        lo = float(ea)
    info = {"scan": scan, "eps_c": float(eps_c)}
    if hi is None:
        info["status"] = "no two-phase point in scan"
        return float("nan"), info
    if lo is None:
        info["status"] = "two-phase at the lowest eps_a scanned"
        return float(hi), info
    for _ in range(refine):
        mid = 0.5 * (lo + hi)
        if two_phase(mid):
            hi = mid
        else:
            lo = mid
    info["status"] = "ok"
    return 0.5 * (lo + hi), info


def solve_curve_robust(eps_a, eps_c, *, phis=None, split_center_vacancy=False,
                       internal_term="min_energy", composition_key="components",
                       tol=1e-8, max_newton=200,
                       accept=1e-6, return_diag=False):
    """
    Free-energy curve with per-phi multi-start.

    A single Newton sweep leaves isolated phi points unconverged on this system
    (~1.5% of points, clustered near the dense branch), and because the binodal is
    read off a convex envelope those holes make the two-phase test flicker: the
    critical line then depends on where the eps_a scan happens to land rather than
    on the physics.  Here every strategy is run over the whole grid and the result
    is merged per phi, keeping the converged one with the lowest residual.

    Strategies: forward and reverse sweeps (so an isolated failure gets a warm
    start from the other side), direct-W and log-W, and cold/hot W seeds.
    """
    phis = np.linspace(0.01, 0.99, 151) if phis is None else np.asarray(phis, float)
    inp = build_inputs(eps_a, eps_c, split_center_vacancy=split_center_vacancy,
                       internal_term=internal_term, composition_key=composition_key)
    Fd = inp[6]["F"]

    strategies = [
        dict(logW=False, W_init=None,                 rev=False),
        dict(logW=False, W_init=None,                 rev=True),
        dict(logW=True,  W_init=None,                 rev=False),
        dict(logW=False, W_init=np.full(Fd, 1e-6),    rev=False),
        dict(logW=False, W_init=np.full(Fd, 1.0),     rev=False),
        dict(logW=True,  W_init=np.full(Fd, 1e-2),    rev=True),
    ]

    best_f = np.full(len(phis), np.nan)
    best_r = np.full(len(phis), np.inf)
    used = np.zeros(len(phis), dtype=np.int64)
    t0 = time.time()
    for k, st in enumerate(strategies):
        order = np.arange(len(phis))[::-1] if st["rev"] else np.arange(len(phis))
        c = solve_free_energy_curve(phis[order], *inp[:5], inp[5],
                                    W_init=st["W_init"], max_newton=max_newton,
                                    tol=tol, accept_residual=accept, logW=st["logW"])
        f = np.empty(len(phis)); f[order] = c["f"]
        r = np.empty(len(phis)); r[order] = c["residual"]
        take = np.isfinite(f) & (r < best_r)
        best_f[take] = f[take]
        best_r[take] = r[take]
        used[take] = k
        if np.all(best_r <= accept):
            break

    ok = best_r <= accept
    best_f[~ok] = np.nan
    # separate the two reasons a phi point can be missing:
    #  - outside [min A_p, max A_p]: the coarse-grained model cannot represent that
    #    composition at all (this is what centre-averaging costs);
    #  - inside the range but Newton did not converge: a numerical failure.
    A = inp[0]
    in_range = (phis >= A.min() - 1e-12) & (phis <= A.max() + 1e-12)
    out = {"phi": phis, "f": best_f, "residual": best_r, "info": inp[6],
           "eps_a": float(eps_a), "eps_c": float(eps_c),
           "solve_time": time.time() - t0,
           "n_unrepresentable": int((~in_range).sum()),
           "n_bad": int((~ok & in_range).sum())}
    if return_diag:
        out["strategy_used"] = used
    return out


def critical_eps_a_lower_edge_robust(eps_c, *, eps_a_scan=None, refine=8, phis=None,
                                     split_center_vacancy=False,
                                     internal_term="min_energy",
                                     composition_key="components",
                                     barrier_tol=ACCEPT_BARRIER,
                                     require_two_spinodal=REQUIRE_TWO_SPINODAL,
                                     coexist_tol=COEXIST_TOL, phi1_max=None,
                                     verbose=False):
    if eps_a_scan is None:
        eps_a_scan = np.arange(0.0, 6.01, 0.25)
    eps_a_scan = np.asarray(eps_a_scan, float)

    def two_phase(ea):
        c = solve_curve_robust(ea, eps_c, phis=phis,
                               split_center_vacancy=split_center_vacancy,
                               internal_term=internal_term,
                               composition_key=composition_key)
        a = analyse_curve(c, coexist_tol=coexist_tol, phi1_max=phi1_max)
        b = a["binodal"]
        ok = b is not None and b["barrier"] > barrier_tol
        if ok and require_two_spinodal:
            ok = len(a["spinodal_phis"]) > 1
        if verbose:
            print(f"    eps_a={ea:6.3f} {ok!s:5s} barrier={(b['barrier'] if b else 0):.2e} "
                  f"nbad={c['n_bad']:2d} rejected={a.get('n_rejected', 0)}", flush=True)
        return ok, c["n_bad"]

    lo = hi = None
    scan = []
    nbad_tot = 0
    for ea in eps_a_scan:
        ok, nb = two_phase(ea)
        nbad_tot += nb
        scan.append((float(ea), bool(ok), int(nb)))
        if ok:
            hi = float(ea)
            break
        lo = float(ea)
    info = {"scan": scan, "eps_c": float(eps_c), "n_bad_total": nbad_tot}
    if hi is None:
        info["status"] = "no two-phase point in scan"
        return float("nan"), info
    if lo is None:
        info["status"] = "two-phase at the lowest eps_a scanned"
        return float(hi), info
    for _ in range(refine):
        mid = 0.5 * (lo + hi)
        ok, nb = two_phase(mid)
        nbad_tot += nb
        if ok:
            hi = mid
        else:
            lo = mid
    info["status"] = "ok"
    info["n_bad_total"] = nbad_tot
    return 0.5 * (lo + hi), info
