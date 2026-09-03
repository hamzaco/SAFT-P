#!/usr/bin/env python
"""
mc_fss_trace.py -- read reference critical points out of the manuscript MC
trace logs (results_logs/all_trace_coexist_{stick,l}.out).

Each continuation step in those logs closes with a complete criticality record:

    === Step: target eps_sp=4.000 (prev=3.980) ===
    ...
    mu* = 5.059693   ESS ~ 5183
    chi^2*  : 0.0142
    theta*  : (0.56607, 4.0, 5.10857, 1.0, 1.19745, 0.29681)
    success : True

theta* = (eps_nd, eps_d, mu, Lambda, M_c, s).  Those six numbers are the
statement "at this (eps_nd, eps_d, mu), with field-mixing parameter s, the
L=10 mixed-field order-parameter distribution matches the 2D-Ising universal
form" -- which is exactly the anchor a finite-size study needs.  The finite-size
run re-asks that question at larger L, starting from this point.

Usage
-----
    python mc_fss_trace.py --system stick --eps-d 4.0
    python mc_fss_trace.py --system l --eps-d 4.0 --window 0.3
    python mc_fss_trace.py --list --system stick
"""

from __future__ import annotations

import argparse
import os
import re

import numpy as np

TRACE = {
    "stick": "all_trace_coexist_stick.out",
    "l": "all_trace_coexist_l.out",
}

_THETA = re.compile(r"^\s*(?:θ\*|theta\*)\s*:\s*\(([^)]*)\)")
_MUSTAR = re.compile(r"^\s*mu\*\s*=\s*([-\d.eE+]+)\s+ESS\s*[≈~=]\s*([-\d.eE+]+)")
_CHI2 = re.compile(r"^\s*(?:χ²\*|chi2\*|chi\^2\*)\s*:\s*([-\d.eE+]+)")
_STEP = re.compile(r"===\s*Step:\s*target\s*eps_sp\s*=\s*([-\d.eE+]+)\s*"
                   r"\(prev\s*=\s*([-\d.eE+]+)\)")
_SUCCESS = re.compile(r"^\s*success\s*:\s*(True|False)")


def parse_trace(path):
    """Return one record per completed continuation step."""
    recs = []
    cur = {}
    with open(path, encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            m = _STEP.search(line)
            if m:
                cur = dict(eps_d_target=float(m.group(1)),
                           eps_d_prev=float(m.group(2)))
                continue
            m = _MUSTAR.match(line)
            if m:
                cur["mu_star_ref"] = float(m.group(1))
                cur["ess_ref"] = float(m.group(2))
                continue
            m = _CHI2.match(line)
            if m:
                cur["js"] = float(m.group(1))
                continue
            m = _SUCCESS.match(line)
            if m:
                cur["success"] = (m.group(1) == "True")
                continue
            m = _THETA.match(line)
            if m:
                parts = [p.strip() for p in m.group(1).split(",")]
                try:
                    v = [float(x) for x in parts[:6]]
                except ValueError:
                    continue
                if len(v) < 6:
                    continue
                rec = dict(cur)
                rec.update(eps_nd=v[0], eps_d=v[1], mu=v[2],
                           lam=v[3], Mc=v[4], s=v[5])
                recs.append(rec)
                cur = {}
    return recs


def clean(recs, s_lo=-1.0, s_hi=1.5):
    """Drop records with an obviously failed field-mixing fit."""
    out = []
    seen = set()
    for r in recs:
        if not np.isfinite([r["eps_nd"], r["mu"], r["s"]]).all():
            continue
        if r["eps_nd"] <= 0 or not (s_lo <= r["s"] <= s_hi):
            continue
        key = (round(r["eps_d"], 6), round(r["eps_nd"], 6), round(r["mu"], 6))
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
    out.sort(key=lambda r: r["eps_d"])
    return out


def reference_point(system, eps_d, trace_dir, window=0.06, smooth_s=True):
    """Nearest trace point to eps_d, with s and mu locally smoothed.

    s scatters step to step (it is refitted from a finite sample each time), so
    the local median over a window in eps_d is a better estimate of the
    field-mixing parameter at eps_d than any single step's value.
    """
    path = os.path.join(trace_dir, TRACE[system])
    recs = clean(parse_trace(path))
    if not recs:
        raise SystemExit(f"no usable records in {path}")
    e = np.array([r["eps_d"] for r in recs])
    i = int(np.argmin(np.abs(e - eps_d)))
    best = dict(recs[i])

    if smooth_s:
        m = np.abs(e - eps_d) <= window
        # The logged sweep has gaps -- the L-shaped trace jumps 3.66 -> 4.18 --
        # so a fixed window can contain nothing, and the raw nearest point then
        # gets used at the wrong eps_d.  For eps_d=4.0 that handed the run the
        # 4.18 anchor, with mu off by 0.17.  Widen until the request is
        # bracketed, then interpolate.
        # Widening by count alone is not enough: around eps_d=4.0 it picks up
        # four points that all sit *above* the gap, and np.interp then clamps
        # to the nearest edge -- silently reproducing the bug it was meant to
        # fix.  Require the request to be bracketed on both sides.
        bracketed = (m.sum() >= 3 and (e[m] < eps_d).any() and (e[m] > eps_d).any())
        if not bracketed and e.min() <= eps_d <= e.max():
            w = float(window)
            lim = float(e.max() - e.min())
            while w < lim:
                w *= 2.0
                m = np.abs(e - eps_d) <= w
                if (m.sum() >= 3 and (e[m] < eps_d).any()
                        and (e[m] > eps_d).any()):
                    break
            best["window_used"] = w
        idx = np.flatnonzero(m)
        order = np.argsort(e[idx])
        idx = idx[order]
        if idx.size >= 3:
            best["s_local_median"] = float(np.median(
                [recs[j]["s"] for j in idx]))
            best["mu_local"] = float(np.interp(
                eps_d, e[idx], [recs[j]["mu"] for j in idx]))
            best["eps_nd_local"] = float(np.interp(
                eps_d, e[idx], [recs[j]["eps_nd"] for j in idx]))
            best["n_local"] = int(idx.size)
            best["interp_gap"] = float(abs(best["eps_d"] - eps_d))
    best["n_records"] = len(recs)
    best["eps_d_available"] = (float(e.min()), float(e.max()))
    return best, recs


def theory_at(system, eps_d, data_dir="../data"):
    """SAFT-P and SAFT critical eps_nd at this eps_d, from the scan outputs.

    Hardcoding these was fine while everything ran at eps_d=4; it silently
    produces the wrong comparison the moment eps_d changes.  SAFT-P is the
    lower envelope of the reduced spinodal scan for that geometry; SAFT is the
    monomer-level boundary, which does not resolve geometry and is therefore
    shared.
    """
    out = {}
    fn = {"stick": "spinodal_stick_shaped_reduced_scan.npz",
          "l": "spinodal_l_shaped_reduced_scan.npz"}.get(system)
    p_sp = os.path.join(data_dir, fn) if fn else None
    if p_sp and os.path.exists(p_sp):
        z = np.load(p_sp)
        pts = z["points"]
        y, x = pts[:, 0], pts[:, 1]
        xq = np.round(x, 6)
        lo = {xv: y[xq == xv].min() for xv in np.unique(xq)}
        xs = np.array(sorted(lo))
        ys = np.array([lo[v] for v in xs])
        out["saft_p"] = float(np.interp(eps_d, ys[::-1], xs[::-1]))

    p_sa = os.path.join(data_dir, "spinodal2_saft.json")
    if os.path.exists(p_sa):
        import json as _json
        with open(p_sa) as fh:
            grid = np.array(_json.load(fh))
        eps_as = np.linspace(0, 10, 21)
        eps_cs = np.linspace(0, 2.5, 21)
        b = [(eps_cs[j], eps_as[np.where(grid[:, j] == 1)[0].min()])
             for j in range(grid.shape[1]) if np.any(grid[:, j] == 1)]
        if b:
            sx = np.array([q[0] for q in b])
            sy = np.array([q[1] for q in b])
            out["saft"] = float(np.interp(eps_d, sy[::-1], sx[::-1]))
    return out


def anchor_at(system, eps_d, trace_dir="../results_logs"):
    """Logged L=10 critical eps_nd at this eps_d (for the DRIFT check)."""
    try:
        b, _ = reference_point(system, eps_d, trace_dir)
        return float(b.get("eps_nd_local", b["eps_nd"]))
    except Exception:                                            # noqa: BLE001
        return None


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--system", default="stick", choices=sorted(TRACE))
    p.add_argument("--eps-d", type=float, default=4.0)
    p.add_argument("--trace-dir", default="../results_logs")
    p.add_argument("--window", type=float, default=0.06)
    p.add_argument("--list", action="store_true")
    args = p.parse_args(argv)

    best, recs = reference_point(args.system, args.eps_d, args.trace_dir,
                                 window=args.window)

    if args.list:
        print(f"{'eps_d':>8} {'eps_nd':>9} {'mu':>9} {'s':>8} {'M_c':>9} "
              f"{'JS':>9} {'ok':>5}")
        for r in recs:
            print(f"{r['eps_d']:>8.3f} {r['eps_nd']:>9.5f} {r['mu']:>9.5f} "
                  f"{r['s']:>8.4f} {r['Mc']:>9.4f} "
                  f"{r.get('js', float('nan')):>9.3e} "
                  f"{str(r.get('success', '?')):>5}")
        return 0

    print(f"system            : {args.system}")
    print(f"records parsed    : {best['n_records']}  "
          f"(eps_d {best['eps_d_available'][0]:.2f} .. "
          f"{best['eps_d_available'][1]:.2f})")
    print(f"nearest to eps_d  = {args.eps_d}")
    print(f"  eps_d           = {best['eps_d']:.5f}")
    print(f"  eps_nd          = {best['eps_nd']:.5f}")
    print(f"  mu              = {best['mu']:.5f}")
    print(f"  s (this step)   = {best['s']:.5f}")
    print(f"  M_c             = {best['Mc']:.5f}")
    if "s_local_median" in best:
        print(f"  s (local median over {best['n_local']} steps) "
              f"= {best['s_local_median']:.5f}")
        print(f"  eps_nd (local interp) = {best['eps_nd_local']:.5f}")
        print(f"  mu     (local interp) = {best['mu_local']:.5f}")
    print()
    print("anchor values (feed to mc_line_sweep.py --eps-nd0/--mu0/--s0):")
    en = best.get("eps_nd_local", best["eps_nd"])
    mu = best.get("mu_local", best["mu"])
    print(f"  --eps-d {args.eps_d} --eps-nd0 {en:.5f} --mu0 {mu:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
