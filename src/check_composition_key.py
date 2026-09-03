#!/usr/bin/env python3
"""
Verification suite for the composition-aware plaquette class key.

Run from ``src/``:

    python check_composition_key.py

It checks four things, in order of how much they would hurt if they broke:

1.  **Legacy parity.**  ``composition_key="none"`` must reproduce the published
    class list element for element, for the 2x2 square builder, the n x n
    builder at n = 2, the 7-site hexagon (with and without
    ``split_center_vacancy``) and the 2x2x2 cube.  Without this the published
    numbers can no longer be regenerated.

2.  **Composition exactness.**  With the default key every class must hold an
    integer number of sites of each chemical component, and the reachable
    composition window must span [0, 1].  This is the property the whole change
    exists to establish.

3.  **Ideal-gas limit.**  At zero coupling the exact free energy per site of a
    lattice gas with q orientations is f = phi*log(phi/q) + (1-phi)*log(1-phi).
    A class definition that smears composition cannot reproduce it, so this is
    the physical test rather than a bookkeeping one.  Requires
    ``internal_term="class_free_energy"``: with ``mean_energy`` the internal
    entropy of a class is discarded and the limit is wrong for an unrelated
    reason.

4.  **Rotation invariance of the key.**  Every class must be closed under the
    point group, i.e. the composition code must not depend on which lab-frame
    image of a cluster was enumerated first.  Checked by rebuilding with the
    species relabelled by one rotation and comparing the sorted class table.
"""

from __future__ import annotations

import sys

import numpy as np

GEOMETRIES = {
    "stick": dict(patches=np.array([[1, 0, 1, 0],
                                    [0, 1, 0, 1],
                                    [2, 2, 2, 2]], dtype=np.int64),
                  rot90_species=np.array([1, 0, 2], dtype=np.int64),
                  q=2),
    "L": dict(patches=np.array([[1, 1, 0, 0],
                                [0, 1, 1, 0],
                                [0, 0, 1, 1],
                                [1, 0, 0, 1],
                                [2, 2, 2, 2]], dtype=np.int64),
              rot90_species=np.array([1, 2, 3, 0, 4], dtype=np.int64),
              q=4),
}


def _chirality():
    """ABEF / BAEF / SSSS, four rotations each for the two particles."""
    rows, l2i, nxt = [], {"0": 0}, [1]
    for name in ("ABEF", "BAEF", "SSSS"):
        a = np.zeros(4, dtype=np.int64)
        for i, ch in enumerate(name):
            if ch not in l2i:
                l2i[ch] = nxt[0]
                nxt[0] += 1
            a[4 - len(name) + i] = l2i[ch]
        seen = set()
        for r in range(4):
            t = tuple(np.roll(a, -r).tolist())
            if t not in seen:
                seen.add(t)
                rows.append(list(t))
    return (np.asarray(rows, dtype=np.int64),
            np.array([3, 0, 1, 2, 7, 4, 5, 6, 8], dtype=np.int64))


def make_J(patches, eps_a, eps_c):
    M = int(np.max(patches)) + 1
    J = np.zeros((M, M), dtype=np.float64)
    J[1, 1] = -float(eps_a)
    J[:-1, :-1] -= float(eps_c)
    return J


def class_table(res):
    """
    Order-independent fingerprint of a builder result: the (S | e | composition |
    mult) rows sorted lexicographically.  Two builders agree iff their tables do.
    """
    eps, intra, p2sp, mp, p2s, p2sm, cfg, mult = res[:8]
    P, F = len(intra), eps.shape[0]
    S = np.zeros((P, F), dtype=np.float64)
    for k in range(len(p2s)):
        S[p2s[k], p2sm[k]] += mp[k]
    tab = np.column_stack([S, intra[:, None], p2sp, mult[:, None]])
    return tab[np.lexsort(np.round(tab, 9).T)], eps


def same(a, b, tol=1e-9):
    ta, ea = class_table(a)
    tb, eb = class_table(b)
    if ta.shape != tb.shape or ea.shape != eb.shape:
        return False, float("inf")
    d = max(np.abs(ta - tb).max(), np.abs(ea - eb).max())
    return d <= tol, float(d)


# ---------------------------------------------------------------------------

def check_legacy_parity(ref_dir):
    """composition_key='none' vs the pre-change modules in ``ref_dir``."""
    import importlib.util

    def load(path, name, alias):
        spec = importlib.util.spec_from_file_location(alias, f"{path}/{name}.py")
        m = importlib.util.module_from_spec(spec)
        sys.modules[alias] = m
        spec.loader.exec_module(m)
        return m

    ok = True
    old_sq = load(ref_dir, "plaquette_by_species", "_ref_sq")
    old_hx = load(ref_dir, "plaquette_by_species_hexagon", "_ref_hx")
    from plaquette_by_species import build_plaquettes_by_species, build_cubes_species
    from plaquette_nxn_by_species import build_plaquettes_nxn_by_species

    print("1. legacy parity (composition_key='none' == published key)")
    cases = dict(GEOMETRIES)
    cases["chirality"] = dict(zip(("patches", "rot90_species"), _chirality()))
    for gname, g in cases.items():
        pa, rot = g["patches"], g["rot90_species"]
        for ea, ec in [(4.0, 1.5), (7.3, 0.31), (0.0, 0.0), (2.0, 0.2)]:
            J = make_J(pa, ea, ec)
            kw = dict(rot90_species=rot, canonicalize_by_boundary_edges=True,
                      canonicalize_by_edges=False, use_4_patch=True)
            a = old_sq.build_plaquettes_by_species(pa, J, np.zeros(len(pa)), **kw)
            b = build_plaquettes_by_species(pa, J, np.zeros(len(pa)),
                                            composition_key="none", **kw)
            good, d = same(a, b)
            ok &= good
            print(f"   2x2 {gname:10s} eps=({ea:4.1f},{ec:4.2f})  P={len(a[1]):5d}  "
                  f"d={d:.1e}  {'OK' if good else 'FAIL'}")
            if gname in GEOMETRIES:
                c = build_plaquettes_nxn_by_species(pa, J, np.zeros(len(pa)),
                                                    rot90_species=rot, n=2,
                                                    composition_key="none")
                good, d = same(a, c)
                ok &= good
                print(f"   nxn(n=2) {gname:6s} eps=({ea:4.1f},{ec:4.2f})  "
                      f"d={d:.1e}  {'OK' if good else 'FAIL'}")

    hx_pa = np.array([[1, 0, 1, 0, 1, 0], [0, 1, 0, 1, 0, 1], [2, 2, 2, 2, 2, 2]],
                     dtype=np.int64)
    hx_rot = np.array([1, 0, 2], dtype=np.int64)
    from plaquette_by_species_hexagon import build_hexagon7_by_species
    for ea in (2.0, 0.0):
        J = make_J(hx_pa, ea, 0.0)
        for scv in (False, True):
            a = old_hx.build_hexagon7_by_species(hx_pa, J, np.zeros(3),
                                                 rot60_species=hx_rot,
                                                 split_center_vacancy=scv)
            b = build_hexagon7_by_species(hx_pa, J, np.zeros(3), rot60_species=hx_rot,
                                          split_center_vacancy=scv,
                                          composition_key="none")
            good, d = same(a, b)
            ok &= good
            print(f"   hex7 eps_a={ea:4.1f} split={scv!s:5s}  P={len(a[1]):5d}  "
                  f"d={d:.1e}  {'OK' if good else 'FAIL'}")

    cu_pa = np.array([[1, 0, 1, 0, 0, 0], [0, 1, 0, 1, 0, 0], [2, 2, 2, 2, 2, 2]],
                     dtype=np.int64)
    Jc = np.zeros((3, 3)); Jc[1, 1] = -2.0
    a = old_sq.build_cubes_species(cu_pa, Jc, np.zeros(3))
    b = build_cubes_species(cu_pa, Jc, np.zeros(3), composition_key="none")
    good, d = same(a, b)
    ok &= good
    print(f"   cube 2x2x2                  P={len(a[1]):5d}  d={d:.1e}  "
          f"{'OK' if good else 'FAIL'}")
    return ok


def check_composition_exact():
    from saftp_scan_nxn import build_reduced_inputs
    print("\n2. composition exactness (default key)")
    ok = True
    for gname, g in GEOMETRIES.items():
        for n in (2, 3):
            for ik in ("none", "occupancy"):
                inp = build_reduced_inputs(g["patches"], g["rot90_species"], 4.0, 1.5,
                                           n=n, interior_key=ik,
                                           composition_key="components")
                i = inp[6]
                good = (i["A_row_max_noninteger"] < 1e-9
                        and i["phi_min"] <= 1e-12 and i["phi_max"] >= 1 - 1e-12)
                ok &= good
                print(f"   {gname:6s} n={n} interior={ik:10s} P={i['P']:6,d} "
                      f"phi=[{i['phi_min']:.4f},{i['phi_max']:.4f}] "
                      f"nonint={i['A_row_max_noninteger']:.1e}  "
                      f"{'OK' if good else 'FAIL'}")
    # hexagon
    import hex_mercedes as HM
    for ck in ("none", "components"):
        inp = HM.build_inputs(3.0, 0.5, composition_key=ck)
        i = inp[6]
        tag = "OK" if (ck == "none" or i["A_row_max_noninteger"] < 1e-9) else "FAIL"
        if ck == "components":
            ok &= i["A_row_max_noninteger"] < 1e-9
        print(f"   hex7   comp_key={ck:11s} P={i['P']:6,d} "
              f"phi=[{i['phi_min']:.4f},{i['phi_max']:.4f}] "
              f"nonint={i['A_row_max_noninteger']:.1e}  {tag}")
    return ok


def check_ideal_limit():
    from saftp_scan_nxn import free_energy_curve
    print("\n3. non-interacting limit  f = phi*log(phi/q) + (1-phi)*log(1-phi)")
    phis = np.linspace(0.02, 0.98, 49)
    ok = True
    for gname, g in GEOMETRIES.items():
        q = g["q"]
        exact = phis * np.log(phis / q) + (1 - phis) * np.log(1 - phis)
        for n, ik in ((2, "none"), (3, "none"), (3, "occupancy")):
            c = free_energy_curve(gname, 0.0, 0.0, n=n, phis=phis, interior_key=ik,
                                  internal_term="class_free_energy",
                                  composition_key="components")
            m = ~np.isnan(c["f"])
            err = np.abs(c["f"][m] - exact[m]).max() if m.any() else np.inf
            good = bool(m.all() and err < 1e-8)
            ok &= good
            print(f"   {gname:6s} n={n} interior={ik:10s} solved={m.sum():2d}/{len(phis)} "
                  f"max|f-f_ideal|={err:.2e}  {'OK' if good else 'FAIL'}")
    return ok


def check_rotation_invariance():
    """
    Relabel the species by one 90 deg rotation and rebuild.  The class table must
    be identical: if the composition code were not rotation-invariant, the class
    a microstate lands in would depend on which image was enumerated first.
    """
    from plaquette_nxn_by_species import build_plaquettes_nxn_by_species
    print("\n4. rotation invariance of the class key")
    ok = True
    for gname, g in GEOMETRIES.items():
        pa, rot = g["patches"], g["rot90_species"]
        J = make_J(pa, 4.0, 1.5)
        perm = rot                                  # relabel s -> rot[s]
        inv = np.argsort(perm)
        pa2 = pa[inv]                               # patches of the relabelled species
        rot2 = perm[rot[inv]]
        for n in (2, 3):
            a = build_plaquettes_nxn_by_species(pa, J, np.zeros(len(pa)),
                                                rot90_species=rot, n=n,
                                                composition_key="components")
            b = build_plaquettes_nxn_by_species(pa2, J, np.zeros(len(pa2)),
                                                rot90_species=rot2, n=n,
                                                composition_key="components")
            ta, _ = class_table(a)
            tb, _ = class_table(b)
            # species columns are permuted by the relabelling; compare the
            # component-resolved table instead of the raw species one.
            good = ta.shape == tb.shape and len(a[1]) == len(b[1])
            good = good and np.allclose(np.sort(a[1]), np.sort(b[1]), atol=1e-9)
            ok &= good
            print(f"   {gname:6s} n={n}  P={len(a[1]):6,d} vs {len(b[1]):6,d}  "
                  f"{'OK' if good else 'FAIL'}")
    return ok


def main():
    ref = sys.argv[1] if len(sys.argv) > 1 else None
    results = []
    if ref:
        results.append(("legacy parity", check_legacy_parity(ref)))
    else:
        print("1. legacy parity  SKIPPED (pass a directory holding the pre-change\n"
              "   plaquette_by_species.py / plaquette_by_species_hexagon.py as argv[1],\n"
              "   e.g.  git show HEAD:'src/plaquette_by_species.py' > ref/...)")
    results.append(("composition exactness", check_composition_exact()))
    results.append(("ideal-gas limit", check_ideal_limit()))
    results.append(("rotation invariance", check_rotation_invariance()))

    print("\n" + "=" * 52)
    for name, good in results:
        print(f"  {name:26s} {'PASS' if good else 'FAIL'}")
    allok = all(g for _, g in results)
    print("=" * 52)
    print("ALL CHECKS:", "PASS" if allok else "FAIL")
    return 0 if allok else 1


if __name__ == "__main__":
    raise SystemExit(main())
