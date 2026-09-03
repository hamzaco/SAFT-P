#!/usr/bin/env python3
"""
Collect the per-temperature SAFT-P binodal JSONs into one table, flag any branch
that is still sitting on the composition-grid endpoint, and compare against the
values used in the published Fig. 7.

    python saftp_binodal_collect.py --workdir /pool/hamza/saftp_binodal
"""
import argparse, glob, json, os

# published Fig. 7 values (phi_lo = 0.01, n_phi = 51)
PUBLISHED = {0.5: (0.010, 0.490), 0.6: (0.010, 0.490), 0.7: (0.010, 0.490),
             0.8: (0.010, 0.490), 0.9: (0.058, 0.442), 1.0: (0.1156, 0.3748),
             1.1: (0.2116, 0.2788)}

p = argparse.ArgumentParser()
p.add_argument("--workdir", required=True)
p.add_argument("--pair", default="BAEF")
p.add_argument("--csv", default=None)
a = p.parse_args()

rows = []
for fn in sorted(glob.glob(os.path.join(a.workdir, f"saftp_T*_{a.pair}.json"))):
    d = json.load(open(fn))
    if d["binodal_phi"] is None:
        rows.append((d["T"], d["T_star"], None, None, None, None, None, d["at_grid_edge"]))
        continue
    p1, p2 = d["binodal_phi"]
    dx = sorted([1 - 4 * p1, 1 - 4 * p2])
    rows.append((d["T"], d["T_star"], p1, p2, dx[0], dx[1],
                 abs(dx[0]) - abs(dx[1]), d["at_grid_edge"]))
rows.sort()

print(f"{'T':>5} {'T*':>5} | {'phi1':>8} {'phi2':>8} | {'Dx(-)':>8} {'Dx(+)':>8} | "
      f"{'asym':>7} | {'edge?':>5} | published Dx")
print("-" * 92)
for T, Ts, p1, p2, dm, dp, asym, edge in rows:
    if p1 is None:
        print(f"{T:5.2f} {Ts:5.2f} |   no binodal found")
        continue
    q = PUBLISHED.get(round(T, 2))
    pub = "" if q is None else f"{min(1-4*q[0],1-4*q[1]):+.3f} / {max(1-4*q[0],1-4*q[1]):+.3f}"
    flag = "YES" if edge else "no"
    print(f"{T:5.2f} {Ts:5.2f} | {p1:8.5f} {p2:8.5f} | {dm:+8.3f} {dp:+8.3f} | "
          f"{asym:+7.3f} | {flag:>5} | {pub}")

if any(r[7] for r in rows if r[2] is not None):
    print("\n!! at least one branch is still on the grid endpoint -- rerun those "
          "temperatures with a smaller --phi-lo before using the numbers.")

if a.csv:
    with open(a.csv, "w") as fh:
        fh.write("T,T_star,phi1,phi2,dx_minus,dx_plus,asymmetry,at_grid_edge\n")
        for r in rows:
            fh.write(",".join("" if x is None else str(x) for x in r) + "\n")
    print(f"\nwrote {a.csv}")
