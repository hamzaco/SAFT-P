"""
Composition-aware class keys for the plaquette / hexagon / cube builders.

Why this module exists
----------------------
All of the cluster builders in this repository group microstates into classes by
their *outgoing boundary signature*, canonicalised over the point group of the
cluster (C4 for the square plaquette, C6 for the compact hexagon, the 24 proper
rotations for the cube).  Two microstates that differ only by a global rotation
are genuinely the same macrostate, so collapsing them is exact.

But the boundary signature does **not** determine the composition of the
cluster.  A site contributes only its outward-facing patch to the boundary; its
species is otherwise invisible, and interior sites are invisible entirely.  So
the plain boundary key merges microstates with *different numbers of particles*
into one class, and the class is then given a Boltzmann-averaged, fractional
composition ``A_p``.  That is not an orientational degeneracy — rotation never
changes how many particles a cluster contains — and it makes the composition
constraint ``sum_p A_p rho_p = phi`` a mean-field smear rather than an exact
bookkeeping identity.  It also leaves the intra-class weights depending on the
chemical potential used at *build* time, which is inconsistent with the mu the
solver later converges to.

The fix is to make the class key composition-aware: a class is
``(boundary signature, composition)`` rather than ``boundary signature`` alone.
Rotational degeneracy is still collapsed exactly; compositional degeneracy is
not collapsed at all.  Then

*  ``A_p`` is an exact integer count over the cluster's sites,
*  every microstate in a class carries the same ``mu . n``, so that factor pulls
   out of the intra-class sum and the class internal free energy is
   mu-independent,
*  and the reachable composition window spans the full ``[0, 1]`` by
   construction.

What counts as "composition"
----------------------------
Not the raw species vector.  In these models the species index labels both the
chemical identity *and* the lattice orientation of a particle: the stick has
species ``{0, 1}`` for one particle in two orientations, the L-shape has
``{0, 1, 2, 3}``, the chirality model has ``{0..3} = ABEE``, ``{4..7} = BAEE``,
``{8..11} = CD``.  A global rotation permutes those labels, so a raw
per-species count vector is *not* rotation-invariant and appending it to the key
would shatter the orientational classes we do want to keep.

The right variable is the count per **chemical component**, where a component is
a set of species closed under the rotation map.  The default component map is
therefore the orbit decomposition of ``rot_species`` itself, which recovers
exactly the intended grouping with no extra input:

    stick      rot90 = [1, 0, 2]           -> {0,1}, {2}            (particle, vacancy)
    L          rot90 = [1, 2, 3, 0, 4]     -> {0,1,2,3}, {4}        (particle, vacancy)
    chirality  rot90 = [1,2,3,0, 5,6,7,4, 9,10,11,8, 12]
                                           -> ABEE, BAEE, CD, vacancy
    cube       (rotations act on patch decorations, not species labels;
                pass rot_map=None -> every species is its own component,
                which is exact because species counts are already invariant)

Because every component is a union of rotation orbits, the component count
vector is invariant under the point group, and the composition code can be
appended to the canonical boundary key without any per-rotation bookkeeping.

Public API
----------
``resolve_component_map``   species -> component id, with validation
``composition_code``        count vector -> single int64 code
``composition_codes_bulk``  vectorised version for (B, S) count blocks
``describe_components``     human-readable summary for notebooks / logs
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "COMPOSITION_KEYS",
    "resolve_component_map",
    "composition_code",
    "composition_codes_bulk",
    "species_counts_bulk",
    "describe_components",
    "check_composition_exactness",
]

#: Accepted values of the ``composition_key`` argument on every builder.
#:
#: ``"none"``        legacy behaviour: boundary signature only.  Classes may mix
#:                   compositions and ``A_p`` is a Boltzmann average.
#: ``"components"``  (default) boundary signature x count per chemical component.
#: ``"species"``     boundary signature x full per-species count vector,
#:                   canonicalised over the point group.  Strictest; separates
#:                   orientational isomer counts as well.
COMPOSITION_KEYS = ("none", "components", "species")


def _orbit_components(rot_map: np.ndarray) -> np.ndarray:
    """
    Orbit decomposition of the permutation ``rot_map``: species s and rot(s) land
    in the same component.  Components are numbered by their smallest member so
    the labelling is deterministic.
    """
    S = int(rot_map.shape[0])
    comp = np.full(S, -1, dtype=np.int64)
    nxt = 0
    for s in range(S):
        if comp[s] >= 0:
            continue
        t = s
        while comp[t] < 0:
            comp[t] = nxt
            t = int(rot_map[t])
        nxt += 1
    return comp


def resolve_component_map(
    n_species: int,
    species_to_component: Any = None,
    *,
    rot_map: Optional[np.ndarray] = None,
    vacancy_index: int = -1,
    rot_name: str = "rot90_species",
) -> Tuple[np.ndarray, int]:
    """
    Resolve the species -> chemical-component map used by the composition key.

    Parameters
    ----------
    n_species
        Number of species (rows of ``patches``).
    species_to_component
        ``None``          use the orbit decomposition of ``rot_map`` (default).
                          With ``rot_map=None`` this degenerates to one component
                          per species, which is the correct strict choice when the
                          point group does not relabel species at all (the cube).
        ``"orbits"``      same as ``None``, stated explicitly.
        ``"occupancy"``   two components: vacancy, and everything else.  Use this
                          to coarsen a multi-component model down to the single
                          occupancy coordinate the reduced solver actually
                          constrains.
        ``"species"``     one component per species (no merging).
        array-like        explicit map of length ``n_species`` with entries in
                          ``[0, C)``; validated for rotation invariance below.
    rot_map
        The species permutation induced by one elementary rotation of the
        cluster (``rot90_species`` / ``rot60_species``).  ``None`` means the
        rotation does not relabel species.
    vacancy_index
        Index of the vacancy/solvent species; only used by ``"occupancy"``.

    Returns
    -------
    (comp_map, n_components)

    Raises
    ------
    ValueError
        If the resolved map is not constant on rotation orbits.  That check is
        the whole safety argument for appending the composition code to the
        canonical key without per-rotation bookkeeping: if a rotation could move
        a species into a different component, the code would not be
        rotation-invariant and the class key would depend on which lab-frame
        image of the cluster happened to be enumerated first.
    """
    S = int(n_species)
    if rot_map is None:
        rot = np.arange(S, dtype=np.int64)
    else:
        rot = np.asarray(rot_map, dtype=np.int64)
        if rot.shape != (S,):
            raise ValueError(f"{rot_name} must have shape ({S},), got {rot.shape}")

    spec = species_to_component
    if spec is None or (isinstance(spec, str) and spec.lower() in ("orbit", "orbits", "auto")):
        comp = _orbit_components(rot)
    elif isinstance(spec, str) and spec.lower() in ("occupancy", "occupied", "binary"):
        vac = int(vacancy_index % S)
        comp = np.ones(S, dtype=np.int64)
        comp[vac] = 0
    elif isinstance(spec, str) and spec.lower() in ("species", "none", "identity"):
        comp = np.arange(S, dtype=np.int64)
    elif isinstance(spec, str):
        raise ValueError(
            f"Unknown species_to_component preset {spec!r}; expected 'orbits', "
            "'occupancy', 'species', or an explicit array."
        )
    else:
        comp = np.asarray(spec, dtype=np.int64)
        if comp.shape != (S,):
            raise ValueError(
                f"species_to_component must have shape ({S},), got {comp.shape}"
            )
        if comp.min() < 0:
            raise ValueError("species_to_component entries must be non-negative.")

    # Renumber densely so the mixed-radix code stays as small as possible.
    _, comp = np.unique(comp, return_inverse=True)
    comp = comp.astype(np.int64)
    C = int(comp.max()) + 1 if comp.size else 0

    bad = np.where(comp[rot] != comp)[0]
    if bad.size:
        raise ValueError(
            "species_to_component is not constant on the orbits of "
            f"{rot_name}: species {bad.tolist()} change component under one "
            "rotation, so the composition code would not be rotation-invariant "
            "and the class key would depend on which lab-frame image happened "
            "to be enumerated first.  Those species are orientations of the "
            "same chemical species -- merge them into one component.  If you "
            "asked for composition_key='species', use 'components' instead: "
            "when the point group relabels species, a rotation class is the "
            "union of the relabelled images of one microstate, so no "
            "per-species count vector can be constant across it and the "
            "per-component count is the finest composition that exists."
        )
    return comp, C


def species_counts_bulk(cfg: np.ndarray, n_species: int) -> np.ndarray:
    """
    (B, S) integer species-count matrix for a (B, n_sites) block of microstates.
    """
    cfg = np.asarray(cfg, dtype=np.int64)
    B, n_sites = cfg.shape
    flat = np.bincount(
        (np.arange(B, dtype=np.int64)[:, None] * n_species + cfg).ravel(),
        minlength=B * n_species,
    )
    return flat.reshape(B, n_species).astype(np.int64, copy=False)


def _radix(n_sites: int, n_components: int) -> Tuple[np.ndarray, int]:
    """
    Mixed-radix place values for a count vector over ``n_components`` bins whose
    entries are bounded by ``n_sites``, plus the total number of codes.
    """
    base = int(n_sites) + 1
    n_codes = base ** int(n_components)
    if n_codes > (1 << 40):
        raise ValueError(
            f"composition code space {n_codes} is too large "
            f"({n_components} components x {n_sites} sites); coarsen "
            "species_to_component (e.g. 'occupancy')."
        )
    place = base ** np.arange(int(n_components), dtype=np.int64)
    return place, int(n_codes)


def composition_code(
    counts: Sequence[float] | np.ndarray,
    comp_map: np.ndarray,
    n_components: int,
    n_sites: int,
) -> Tuple[int, int]:
    """
    Encode one species-count vector as a single integer composition code.

    Returns ``(code, n_codes)`` so callers can combine it with a boundary code
    as ``boundary * n_codes + code`` without recomputing the radix.
    """
    counts = np.asarray(counts, dtype=np.int64)
    place, n_codes = _radix(n_sites, n_components)
    comp_counts = np.bincount(comp_map, weights=counts, minlength=n_components)
    return int(np.dot(comp_counts.astype(np.int64), place)), n_codes


def composition_codes_bulk(
    counts: np.ndarray,
    comp_map: np.ndarray,
    n_components: int,
    n_sites: int,
) -> Tuple[np.ndarray, int]:
    """
    Vectorised :func:`composition_code` for a (B, S) block of count vectors.
    """
    counts = np.asarray(counts, dtype=np.int64)
    place, n_codes = _radix(n_sites, n_components)
    B = counts.shape[0]
    comp_counts = np.zeros((B, n_components), dtype=np.int64)
    np.add.at(comp_counts.T, comp_map, counts.T)
    return comp_counts @ place, n_codes


def describe_components(comp_map: np.ndarray, n_components: int) -> str:
    """One-line summary of the component map, for notebook output and logs."""
    comp_map = np.asarray(comp_map, dtype=np.int64)
    groups = [np.where(comp_map == c)[0].tolist() for c in range(n_components)]
    return " | ".join(f"c{c}={g}" for c, g in enumerate(groups))


def check_composition_exactness(
    plaq_to_species: np.ndarray,
    comp_map: np.ndarray,
    n_components: int,
    n_sites: int,
    *,
    atol: float = 1e-9,
) -> Dict[str, Any]:
    """
    Verify that every class has an integer number of sites in each chemical
    component -- the property the composition key is supposed to guarantee.

    Returns a dict with the worst deviation and the per-class component counts,
    so notebooks can assert on it instead of eyeballing ``A_p``.
    """
    p2s = np.asarray(plaq_to_species, dtype=np.float64)
    comp_map = np.asarray(comp_map, dtype=np.int64)
    P = p2s.shape[0]
    comp_counts = np.zeros((P, n_components), dtype=np.float64)
    np.add.at(comp_counts.T, comp_map, p2s.T)
    resid = np.abs(comp_counts - np.rint(comp_counts))
    return {
        "component_counts": comp_counts,
        "max_noninteger_residual": float(resid.max()) if resid.size else 0.0,
        "is_exact": bool(resid.max() <= atol) if resid.size else True,
        "n_sites_check": float(np.abs(comp_counts.sum(axis=1) - n_sites).max())
        if comp_counts.size
        else 0.0,
    }
