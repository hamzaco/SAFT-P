"""
n x n square plaquettes grouped by boundary super-edge signature.

This is the direct generalisation of ``build_plaquettes_by_species`` in
``plaquette_by_species.py`` from the 2x2 plaquette to an arbitrary n x n
plaquette (n = 2 gives results identical to the original routine, up to the
ordering of the class list).

Definitions, all identical in spirit to the 2x2 construction
-----------------------------------------------------------
*  A microstate is an assignment of one species to each of the n^2 sites.
*  The outgoing boundary of the plaquette is read clockwise, starting at the
   top-left corner:

       top    = (N patch of row 0, left -> right)
       right  = (E patch of col n-1, top -> bottom)
       bottom = (S patch of row n-1, right -> left)
       left   = (W patch of col 0,  bottom -> top)

   Each side is therefore an ordered n-tuple of patch ids, encoded as a single
   "super-edge" id in base M (M = number of patch types).
*  Two microstates are in the same class iff their 4-tuple of super-edge ids
   agree up to a cyclic shift (i.e. up to a global C4 rotation of the
   plaquette).  Because the species rotation map satisfies
   ``patches[rot(s)][N] == patches[s][W]`` etc., rotating the microstate and
   recomputing the boundary is exactly a cyclic shift of the 4-tuple, so the
   class key can be computed without ever rotating a configuration.
*  Within a class every microstate quantity (internal energy, species counts,
   per-super-edge-type counts) is Boltzmann-averaged with weight
   ``exp(-(E_int - mu.n))``, and the class multiplicity is set to 1 because the
   degeneracy already sits inside that weight.  This mirrors the 2x2 code
   exactly (``mult_arr.append(1.0)`` under boundary-edge canonicalisation).

Two adjacent plaquettes meet along a full side.  Super-edge i = (a_0,...,a_{n-1})
faces super-edge j = (b_0,...,b_{n-1}) with the slot order reversed by the
clockwise convention, so the interface energy is

    eps_small[i, j] = sum_k J[a_k, b_{n-1-k}]

which for n = 2 reduces to ``J[a0,b1] + J[a1,b0]``, the expression used in the
2x2 code.

Enumeration is fully vectorised: all S^(n^2) microstates are built as one array
of digits and reduced with ``np.bincount`` / ``reduceat``.  For the two systems
in the manuscript this is 3^9 = 19,683 (stick) and 5^9 = 1,953,125 (L-shaped).
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np

from plaquette_composition import (
    COMPOSITION_KEYS,
    resolve_component_map,
    composition_codes_bulk,
    describe_components,
    species_counts_bulk,
)

N_DIR, E_DIR, S_DIR, W_DIR = 0, 1, 2, 3


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _rot90_species_map(n_species: int, rot90_species: Optional[np.ndarray]) -> np.ndarray:
    if rot90_species is None:
        return np.arange(n_species, dtype=np.int64)
    r = np.asarray(rot90_species, dtype=np.int64)
    if r.shape != (n_species,):
        raise ValueError(f"rot90_species must have shape ({n_species},), got {r.shape}")
    if np.unique(r).size != n_species:
        raise ValueError("rot90_species must be a permutation.")
    return r


def check_rot90_consistency(patches: np.ndarray, rot90_species: Optional[np.ndarray]) -> None:
    """
    Assert that the species rotation map really implements a 90 deg CW rotation
    of the patch decoration:  N <- W, E <- N, S <- E, W <- S.

    This is what lets the class key be computed as a cyclic shift of the
    boundary 4-tuple instead of by re-rotating every microstate.
    """
    patches = np.asarray(patches, dtype=np.int64)
    S = patches.shape[0]
    rot = _rot90_species_map(S, rot90_species)
    got = patches[rot]                       # patches of the rotated species
    want = patches[:, [W_DIR, N_DIR, E_DIR, S_DIR]]
    if not np.array_equal(got, want):
        bad = np.where(np.any(got != want, axis=1))[0]
        raise ValueError(
            "rot90_species is not consistent with `patches` under a 90 deg CW "
            f"rotation (N<-W, E<-N, S<-E, W<-S).  Offending species: {bad.tolist()}"
        )


def _digits(n_states: int, n_sites: int, base: int) -> np.ndarray:
    """
    (n_states, n_sites) int8 array of base-`base` digits, most significant digit
    in column 0.  Row k is the microstate with index k.
    """
    idx = np.arange(n_states, dtype=np.int64)
    out = np.empty((n_states, n_sites), dtype=np.int8)
    for j in range(n_sites - 1, -1, -1):
        out[:, j] = (idx % base).astype(np.int8)
        idx //= base
    return out


# ---------------------------------------------------------------------------
# builder
# ---------------------------------------------------------------------------

def build_plaquettes_nxn_by_species(
    patches: np.ndarray,
    J: np.ndarray,
    mu: Optional[np.ndarray] = None,
    rot90_species: Optional[np.ndarray] = None,
    *,
    n: int = 3,
    interior_key: str = "occupancy",
    internal_term: str = "mean_energy",
    composition_key: str = "components",
    species_to_component: Any = None,
    vacancy_index: int = -1,
    undirected_edges: bool = False,
    return_diagnostics: bool = False,
    check_rotation_map: bool = True,
) -> Any:
    """
    Returns the same 8-tuple as ``build_plaquettes_by_species(..., use_4_patch=True,
    canonicalize_by_boundary_edges=True)``:

        eps_small, intra_bonds, plaq_to_species, m_patch,
        patch_to_species, patch_to_small, plaq_configs, mult

    with ``plaq_configs`` now of shape (P, n*n) in row-major site order.

    interior_key
        What, besides the boundary super-edge signature, labels a class.  For
        n = 2 there are no interior sites and the option is inert; for n >= 3 it
        decides how the (n-2)^2 hidden sites are treated.

        "none"       Pure boundary signature -- the literal generalisation of
                     the published 2x2 recipe.  The hidden sites are
                     Boltzmann-averaged at zero chemical potential, so their
                     occupancy is frozen at its ideal value and the reachable
                     composition window shrinks to
                     phi in [(S-1)/(S*n^2) * n_int, 1 - ...] rather than [0, 1].
                     Cheapest, but it cannot represent a pure solvent phase.

        "occupancy"  (default) Boundary signature x occupancy pattern of the
                     interior sites (canonicalised together under C4).  The
                     class composition is then exact, phi spans [0, 1], and the
                     interior *orientational* freedom is still Boltzmann-
                     averaged -- which is legitimate, because orientation
                     carries no composition.  For n = 3 this at most doubles
                     the class count.

        "species"    Boundary signature x full interior species pattern.  No
                     averaging over hidden sites at all; largest class count.

    composition_key
        Whether the class key also carries the composition of the plaquette.

        The boundary super-edge signature does not determine what the plaquette
        is made of: a perimeter site shows only its outward patch and the
        interior is invisible, so microstates with different particle counts
        share a signature and used to be merged into one class with a
        fractional, Boltzmann-averaged ``A_p``.  A global rotation is a symmetry
        and may be collapsed exactly; a change of composition is not, and the
        two must not be conflated.

        "components"  (default) class key = boundary signature x interior label
                      x count per chemical component.  A component is a set of
                      species closed under ``rot90_species``; the default map is
                      the orbit decomposition of that permutation, so the
                      stick's {0,1} and the L-shape's {0,1,2,3} merge into one
                      particle component while the vacancy stays separate.
                      Component counts are C4-invariant, so the code multiplies
                      into the canonical key without extra bookkeeping.  With
                      this on, ``A_p`` is an exact multiple of 1/n^2, the
                      reachable phi spans [0, 1] for any ``interior_key``, and
                      ``mu . n`` is a class constant, so the class internal free
                      energy no longer depends on the mu used at build time.

        "species"     one component per species; only well defined when
                      ``rot90_species`` is the identity.

        "none"        legacy behaviour: boundary signature x interior label
                      only.  Kept for regression against the published numbers.

        Note that ``interior_key="occupancy"`` is a weaker, n-specific version
        of the same idea: it pins the occupancy *pattern* of the hidden sites,
        whereas ``composition_key`` pins the *total* composition of the whole
        plaquette.  With ``composition_key="components"`` the composition is
        exact even at ``interior_key="none"``.

    internal_term
        What goes into ``intra_bonds`` (the per-class term the solver puts in
        the exponent alongside ``log_mult``).

        "mean_energy"        <E>_p, the Boltzmann-averaged internal energy, with
                             mult = 1.  This is exactly what the published 2x2
                             code does.  The internal entropy of a class is
                             discarded; at 2x2 that costs little, but the number
                             of microstates per class grows quickly with n, so
                             for a 2x2 vs 3x3 comparison it is worth checking
                             both.

        "class_free_energy"  g_p = -log Z_p with Z_p = sum_micro exp(-E_eff),
                             again with mult = 1.  The class then carries its
                             full internal partition function, which is the
                             thermodynamically consistent coarse-graining.

    vacancy_index
        Index of the vacancy/solvent species (default: last).  Used only by
        ``interior_key="occupancy"``.
    """
    patches = np.asarray(patches, dtype=np.int64)
    J = np.asarray(J, dtype=np.float64)

    S = int(patches.shape[0])
    M = int(J.shape[0])
    if patches.shape[1] != 4:
        raise ValueError("patches must have shape (S, 4) in (N, E, S, W) order.")
    if J.shape != (M, M):
        raise ValueError(f"J must be square, got {J.shape}")
    if patches.max() >= M or patches.min() < 0:
        raise ValueError("patch ids must lie in [0, M).")

    mu_vec = np.zeros(S, dtype=np.float64) if mu is None else np.asarray(mu, dtype=np.float64)
    if mu_vec.shape != (S,):
        raise ValueError(f"mu must have shape ({S},), got {mu_vec.shape}")

    if check_rotation_map:
        check_rot90_consistency(patches, rot90_species)

    composition_key = str(composition_key).strip().lower()
    if composition_key not in COMPOSITION_KEYS:
        raise ValueError(f"composition_key must be one of {COMPOSITION_KEYS}, got {composition_key!r}")

    rot_species_map = _rot90_species_map(S, rot90_species)
    if composition_key == "none":
        comp_map = np.arange(S, dtype=np.int64)
        n_components = S
    else:
        comp_map, n_components = resolve_component_map(
            S,
            "species" if composition_key == "species" else species_to_component,
            rot_map=rot_species_map,
            vacancy_index=vacancy_index,
            rot_name="rot90_species",
        )
        members_by_comp = [np.where(comp_map == c)[0] for c in range(n_components)]
        for c, members in enumerate(members_by_comp):
            if members.size > 1 and not np.allclose(mu_vec[members], mu_vec[members[0]]):
                import warnings
                warnings.warn(
                    f"mu is not constant on chemical component {c} "
                    f"(species {members.tolist()}); with composition-aware "
                    "classes mu.n is meant to be a class constant, so the "
                    "class internal free energy will keep a residual "
                    "dependence on the build-time mu.",
                    RuntimeWarning,
                )

    n_sites = n * n
    n_states = S ** n_sites
    if n_states > 40_000_000:
        raise MemoryError(
            f"S**(n*n) = {n_states:,} microstates is too large for exhaustive "
            "enumeration in this routine."
        )

    pN = patches[:, N_DIR]
    pE = patches[:, E_DIR]
    pS = patches[:, S_DIR]
    pW = patches[:, W_DIR]

    cfg = _digits(n_states, n_sites, S)          # (n_states, n_sites) int8
    site = lambda r, c: r * n + c                # noqa: E731

    # ---- internal energy ---------------------------------------------------
    Eint = np.zeros(n_states, dtype=np.float64)
    for r in range(n):
        for c in range(n - 1):                   # horizontal bonds
            a = cfg[:, site(r, c)]
            b = cfg[:, site(r, c + 1)]
            Eint += J[pE[a], pW[b]]
    for r in range(n - 1):
        for c in range(n):                       # vertical bonds
            a = cfg[:, site(r, c)]
            b = cfg[:, site(r + 1, c)]
            Eint += J[pS[a], pN[b]]

    Eeff = Eint.copy()
    if np.any(mu_vec != 0.0):
        mu_cfg = np.zeros(n_states, dtype=np.float64)
        for j in range(n_sites):
            mu_cfg += mu_vec[cfg[:, j]]
        Eeff -= mu_cfg

    # ---- boundary super-edges (clockwise from the top-left corner) ---------
    powM = M ** np.arange(n - 1, -1, -1, dtype=np.int64)   # base-M place values

    def _pack(patch_cols) -> np.ndarray:
        acc = np.zeros(n_states, dtype=np.int64)
        for k, col in enumerate(patch_cols):
            acc += col.astype(np.int64) * powM[k]
        return acc

    top = _pack([pN[cfg[:, site(0, c)]] for c in range(n)])
    right = _pack([pE[cfg[:, site(r, n - 1)]] for r in range(n)])
    bottom = _pack([pS[cfg[:, site(n - 1, c)]] for c in range(n - 1, -1, -1)])
    left = _pack([pW[cfg[:, site(r, 0)]] for r in range(n - 1, -1, -1)])

    if undirected_edges:
        # optional: identify a super-edge with its reversal
        def _rev(x):
            d = np.stack([(x // powM[k]) % M for k in range(n)], axis=1)
            return _pack([d[:, n - 1 - k] for k in range(n)])
        for arr in (top, right, bottom, left):
            arr[:] = np.minimum(arr, _rev(arr))

    sides = np.stack([top, right, bottom, left], axis=1)   # (n_states, 4)

    # ---- interior label ----------------------------------------------------
    # Interior sites are those not on the perimeter.  Under a 90 deg CW rotation
    # of the plaquette the interior sub-grid rotates the same way, so the label
    # must be rotated in lock-step with the cyclic shift of the boundary tuple.
    if interior_key not in ("none", "occupancy", "species"):
        raise ValueError("interior_key must be 'none', 'occupancy' or 'species'.")

    n_int = max(n - 2, 0)
    interior_sites = [[site(r, c) for c in range(1, n - 1)] for r in range(1, n - 1)]
    vac = int(vacancy_index % S)

    if n_int == 0 or interior_key == "none":
        int_codes = [np.zeros(n_states, dtype=np.int64)] * 4
        n_int_codes = 1
    else:
        if interior_key == "occupancy":
            base = 2
            val = (cfg[:, [s for row in interior_sites for s in row]] != vac).astype(np.int64)
            rot_val = lambda v: v                          # noqa: E731  occupancy is rot-invariant
        else:
            base = S
            val = cfg[:, [s for row in interior_sites for s in row]].astype(np.int64)
            rot = _rot90_species_map(S, rot90_species)
            rot_val = lambda v: rot[v]                     # noqa: E731
        val = val.reshape(n_states, n_int, n_int)
        n_int_codes = base ** (n_int * n_int)
        place = base ** np.arange(n_int * n_int - 1, -1, -1, dtype=np.int64)
        int_codes = []
        cur = val
        for k in range(4):
            flat_ = cur.reshape(n_states, -1)
            int_codes.append((flat_ * place[None, :]).sum(axis=1))
            # rotate the interior sub-grid 90 deg CW: new[r,c] = old[n_int-1-c, r]
            cur = rot_val(np.transpose(cur, (0, 2, 1))[:, :, ::-1])

    # ---- composition label -------------------------------------------------
    # Chemical-component counts are invariant under C4 (every component is a
    # union of rotation orbits), so the code is the same for all four images and
    # multiplies into the key after the minimisation rather than inside it.
    if composition_key == "none":
        comp_codes = np.zeros(n_states, dtype=np.int64)
        n_comp_codes = 1
    else:
        comp_codes, n_comp_codes = composition_codes_bulk(
            species_counts_bulk(cfg, S), comp_map, n_components, n_sites)

    # class key: minimum over the four C4 images, then the composition label
    NE = M ** n                                            # number of super-edge ids
    max_key = (NE ** 4) * n_int_codes * n_comp_codes
    if max_key > np.iinfo(np.int64).max // 4:
        raise ValueError(
            f"class-key code space {max_key} overflows int64; coarsen "
            "species_to_component or interior_key."
        )
    shifts = np.empty((n_states, 4), dtype=np.int64)
    for k in range(4):
        idx = [(j - k) % 4 for j in range(4)]              # rotate: (t,r,b,l) -> (l,t,r,b)
        s = sides[:, idx]
        code = ((s[:, 0] * NE + s[:, 1]) * NE + s[:, 2]) * NE + s[:, 3]
        shifts[:, k] = code * n_int_codes + int_codes[k]
    key = shifts.min(axis=1) * n_comp_codes + comp_codes

    uniq_key, gid = np.unique(key, return_inverse=True)
    P = int(uniq_key.size)
    gid = gid.astype(np.int64, copy=False)

    # ---- Boltzmann averaging inside each class ----------------------------
    order = np.argsort(gid, kind="stable")
    gid_s = gid[order]
    Eeff_s = Eeff[order]
    starts = np.searchsorted(gid_s, np.arange(P), side="left")

    E0 = np.minimum.reduceat(Eeff_s, starts)               # per-class energy shift
    argmin_local = np.empty(P, dtype=np.int64)
    ends = np.append(starts[1:], n_states)
    for i in range(P):
        seg = Eeff_s[starts[i]:ends[i]]
        argmin_local[i] = order[starts[i] + int(np.argmin(seg))]
    best_cfg_idx = argmin_local

    w = np.exp(-(Eeff - E0[gid]))
    w_sum = np.bincount(gid, weights=w, minlength=P)
    Eeff_sum = np.bincount(gid, weights=w * Eeff, minlength=P)

    species_sum = np.zeros((P, S), dtype=np.float64)
    flat = np.zeros(P * S, dtype=np.float64)
    for j in range(n_sites):
        flat += np.bincount(gid * S + cfg[:, j].astype(np.int64), weights=w, minlength=P * S)
    species_sum = flat.reshape(P, S)

    boundary_sum = np.zeros(P * NE, dtype=np.float64)
    for k in range(4):
        boundary_sum += np.bincount(gid * NE + sides[:, k], weights=w, minlength=P * NE)
    boundary_sum = boundary_sum.reshape(P, NE)

    if internal_term not in ("mean_energy", "class_free_energy"):
        raise ValueError("internal_term must be 'mean_energy' or 'class_free_energy'.")

    inv_w = 1.0 / w_sum
    class_logZ = -E0 + np.log(w_sum)                 # log sum_micro exp(-E_eff)
    if internal_term == "mean_energy":
        intra_bonds = Eeff_sum * inv_w
    else:
        intra_bonds = -class_logZ
    plaq_to_species = species_sum * inv_w[:, None]
    boundary_avg = boundary_sum * inv_w[:, None]
    plaq_configs = cfg[best_cfg_idx].astype(np.int64)
    mult = np.ones(P, dtype=np.float64)

    # ---- super-edge type table --------------------------------------------
    used = np.where(boundary_avg.sum(axis=0) > 0.0)[0]
    unique_edge_ids = used.astype(np.int64)
    Fd = int(unique_edge_ids.size)
    boundary_avg = boundary_avg[:, used]

    rows, cols = np.nonzero(boundary_avg)
    patch_to_species = rows.astype(np.int64)
    patch_to_small = cols.astype(np.int64)
    m_patch = boundary_avg[rows, cols].astype(np.float64)

    slots = np.stack([(unique_edge_ids // powM[k]) % M for k in range(n)], axis=1)  # (Fd, n)
    eps_small = np.zeros((Fd, Fd), dtype=np.float64)
    for k in range(n):
        eps_small += J[slots[:, k][:, None], slots[:, n - 1 - k][None, :]]

    if return_diagnostics:
        comp_counts = np.zeros((P, n_components), dtype=np.float64)
        np.add.at(comp_counts.T, comp_map, plaq_to_species.T)
        resid = np.abs(comp_counts - np.rint(comp_counts))
        diag: Dict[str, Any] = {
            "n": n,
            "n_sites": n_sites,
            "n_microstates": n_states,
            "n_classes": P,
            "composition_key": composition_key,
            "comp_map": comp_map,
            "n_components": n_components,
            "components": describe_components(comp_map, n_components),
            "component_counts": comp_counts,
            # 0 by construction whenever composition_key != "none".
            "max_noninteger_residual": float(resid.max()) if resid.size else 0.0,
            "n_super_edge_types": Fd,
            "unique_edge_ids": unique_edge_ids,
            "edge_slots": slots,
            "class_size": np.bincount(gid, minlength=P),
            "class_logZ": class_logZ,            # log sum_micro exp(-Eeff)
            "mean_energy": Eeff_sum * inv_w,
            "E0": E0,
            "boundary_avg": boundary_avg,
        }
        return (eps_small, intra_bonds, plaq_to_species, m_patch,
                patch_to_species, patch_to_small, plaq_configs, mult, diag)

    return (eps_small, intra_bonds, plaq_to_species, m_patch,
            patch_to_species, patch_to_small, plaq_configs, mult)


def count_classes(patches, rot90_species, *, n=3) -> Tuple[int, int, int]:
    """
    (n_microstates, n_distinct_boundary_4tuples, n_classes_after_C4).
    Purely combinatorial: no energies involved.
    """
    patches = np.asarray(patches, dtype=np.int64)
    S, M = int(patches.shape[0]), int(patches.max()) + 1
    n_sites, n_states = n * n, S ** (n * n)
    cfg = _digits(n_states, n_sites, S)
    pN, pE, pS, pW = patches[:, 0], patches[:, 1], patches[:, 2], patches[:, 3]
    powM = M ** np.arange(n - 1, -1, -1, dtype=np.int64)
    site = lambda r, c: r * n + c  # noqa: E731

    def _pack(cols):
        acc = np.zeros(n_states, dtype=np.int64)
        for k, col in enumerate(cols):
            acc += col.astype(np.int64) * powM[k]
        return acc

    sides = np.stack([
        _pack([pN[cfg[:, site(0, c)]] for c in range(n)]),
        _pack([pE[cfg[:, site(r, n - 1)]] for r in range(n)]),
        _pack([pS[cfg[:, site(n - 1, c)]] for c in range(n - 1, -1, -1)]),
        _pack([pW[cfg[:, site(r, 0)]] for r in range(n - 1, -1, -1)]),
    ], axis=1)

    NE = M ** n
    raw = ((sides[:, 0] * NE + sides[:, 1]) * NE + sides[:, 2]) * NE + sides[:, 3]
    n_raw = int(np.unique(raw).size)

    shifts = np.empty((n_states, 4), dtype=np.int64)
    for k in range(4):
        s = sides[:, [(j - k) % 4 for j in range(4)]]
        shifts[:, k] = ((s[:, 0] * NE + s[:, 1]) * NE + s[:, 2]) * NE + s[:, 3]
    n_canon = int(np.unique(shifts.min(axis=1)).size)
    return n_states, n_raw, n_canon
