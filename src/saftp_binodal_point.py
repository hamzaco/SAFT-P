#!/usr/bin/env python3
"""
One (temperature) point of the SAFT-P geometric-isomer binodal scan.

    python saftp_binodal_point.py --T 0.7 --n-phi 201 --phi-lo 0.002 --outdir /pool/hamza/saftp_binodal

Writes <outdir>/saftp_T<T>.json containing the binodal, the spinodal and the full
f(phi) curve, so the common-tangent construction can be redone offline without
re-running the optimisation.

Background: the published Fig. 7 used phi_lo=0.01, n_phi=51.  For T <= 0.8 the
common-tangent construction then returned phi1=0.01 and phi2=0.49 -- exactly the
grid endpoints -- so those branches were grid-limited rather than converged.
"""
import argparse, json, os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from saftp_chirality import *   # noqa: F403  (brings in plaquette_by_species too)

p = argparse.ArgumentParser()
p.add_argument("--T", type=float, required=True)
p.add_argument("--n-phi", type=int, default=201)
p.add_argument("--phi-lo", type=float, default=0.002)
p.add_argument("--rho-empty", type=float, default=0.5)
p.add_argument("--pair", default="BAEF", help="second species; 'AFEB' gives the enantiomer control")
p.add_argument("--outdir", default=".")
p.add_argument("--composition-key", default="components", choices=["components", "species", "none"],
               help="Plaquette class key.  'components' (default) keys a class on "
                    "(boundary signature, per-chemical-component monomer counts), so a class "
                    "cannot mix ABEF-, BAEF- and solvent-rich microstates behind one boundary; "
                    "'none' reproduces the published boundary-only key.")
a = p.parse_args()

os.makedirs(a.outdir, exist_ok=True)
species = get_species_list_ind(["ABEF", a.pair, "SSSS"])
patches = np.array([s.patches for s in species])
rot90 = np.array([3, 0, 1, 2, 7, 4, 5, 6, 8], dtype=np.int64)
n_pt = patches.max() + 1
J = np.zeros((n_pt, n_pt))
J[1, 3] = -3.0        # A-E
J[2, 4] = -3.0        # B-F
J[3, 3] = -1.0        # E-E   (note: J+J.T below doubles the diagonal terms)
J[4, 4] = -1.0        # F-F
J = J + J.T

t0 = time.time()
spin, bounds, bino = find_spinodal_chirality_species(
    patches, [a.T], 4,
    rho_empty=a.rho_empty, z=4, n_phi=a.n_phi, phi_lo=a.phi_lo,
    lr=0.5, tol=1e-5, rot90_species=rot90, use_4_patch=True, J_template=J,
    canonicalize_by_boundary_edges=True, canonicalize_by_edges=False,
    composition_key=a.composition_key,
)
b = bino[0] if isinstance(bino[0], dict) else None
s = spin[0]
out = {
    "T": a.T, "T_star": a.T / 2.0, "pair": a.pair,
    "n_phi": a.n_phi, "phi_lo": a.phi_lo, "rho_empty": a.rho_empty,
    "composition_key": a.composition_key,
    "seconds": time.time() - t0,
    "binodal_phi": None if b is None else [b["phi1"], b["phi2"]],
    "binodal_dx": None if b is None else [1 - 4 * b["phi1"], 1 - 4 * b["phi2"]],
    "binodal_barrier": None if b is None else b["barrier"],
    "at_grid_edge": None if b is None else bool(
        abs(b["phi1"] - a.phi_lo) < 1e-9 or abs(b["phi2"] - (1 - a.rho_empty - a.phi_lo)) < 1e-9),
    "spinodal_phi": None if s is None else np.asarray(s["spinodal_phis"], float).tolist(),
    "phi": None if s is None else np.asarray(s["phi"], float).tolist(),
    "f": None if s is None else np.asarray(s["f"], float).tolist(),
}
fn = os.path.join(a.outdir, f"saftp_T{a.T:.2f}_{a.pair}.json")
with open(fn, "w") as fh:
    json.dump(out, fh)
print(f"T={a.T}  binodal_phi={out['binodal_phi']}  at_grid_edge={out['at_grid_edge']}  "
      f"({out['seconds']:.0f} s) -> {fn}", flush=True)
