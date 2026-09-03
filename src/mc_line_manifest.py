#!/usr/bin/env python
"""
mc_line_manifest.py -- cut the critical-line sweep into segments and give each
one an anchor (eps_nd, mu, s) taken from the logged L=10 continuation.

Why segments
------------
The notebook's sweep is a single sequential continuation: step i+1 starts from
the state located at step i, so it cannot be parallelised as written.  It can
be *restarted*, though -- that is exactly what the notebook cells
`main(1.1039..., 4.18, 6.6654)` do, and the manuscript line was produced that
way.  So: cut [eps_d_min, eps_d_max] into segments, and start each one from the
logged L=10 criticality record at its left edge.  Every segment is then an
independent SLURM task.

The cost of that is honest and should be stated in the paper: at L>10 each
segment inherits the L=10 anchor at its left edge, so a segment reports how the
line *moves* away from the L=10 line over its own width, not an absolute point
walked in from eps_d=0.  Segment-to-segment agreement in the overlap (see
--overlap) is the check that this is harmless; the first segment, which starts
at the exact lattice-gas point at eps_d=0, is anchor-free at every L.

Output format (one line per task, consumed by run_mc_line.sh):

    SYSTEM  L  SEG  EPS_D_START  EPS_D_END  EPS_ND0  MU0  S0  SEED

EPS_ND0/MU0/S0 are '-' for the segment starting at eps_d=0, where
mc_line_sweep.py uses the exact lattice-gas anchor instead.

Usage
-----
    python mc_line_manifest.py --system l --L 20 --eps-d-min 0 --eps-d-max 6 \
        --width 0.2 --seeds "1 2 3"
    python mc_line_manifest.py --system l --L 30 --width auto --list-anchors
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# The trace parser already exists and is exercised by the fixed-eps_d campaign;
# re-using it keeps one definition of "what the logs say".  A local fallback
# keeps this script usable if mc_fss_trace.py is ever moved.
try:
    from mc_fss_trace import parse_trace, clean, TRACE
except Exception:                                              # noqa: BLE001
    import re

    TRACE = {"stick": "all_trace_coexist_stick.out",
             "l": "all_trace_coexist_l.out"}
    _THETA = re.compile(r"^\s*(?:θ\*|theta\*)\s*:\s*\(([^)]*)\)")

    def parse_trace(path):                                     # noqa: D103
        recs = []
        with open(path, encoding="utf-8", errors="ignore") as fh:
            for line in fh:
                m = _THETA.match(line)
                if not m:
                    continue
                parts = [p.strip() for p in m.group(1).split(",")]
                try:
                    v = [float(x) for x in parts[:6]]
                except ValueError:
                    continue
                if len(v) < 6:
                    continue
                recs.append(dict(eps_nd=v[0], eps_d=v[1], mu=v[2],
                                 lam=v[3], Mc=v[4], s=v[5]))
        return recs

    def clean(recs, s_lo=-1.0, s_hi=1.5):                      # noqa: D103
        out, seen = [], set()
        for r in recs:
            if not np.isfinite([r["eps_nd"], r["mu"], r["s"]]).all():
                continue
            if r["eps_nd"] <= 0 or not (s_lo <= r["s"] <= s_hi):
                continue
            key = (round(r["eps_d"], 6), round(r["eps_nd"], 6),
                   round(r["mu"], 6))
            if key in seen:
                continue
            seen.add(key)
            out.append(r)
        out.sort(key=lambda r: r["eps_d"])
        return out


# Default segment width per L.  Chosen so one task is a few hours to ~a day:
# cost per eps_d step scales as (L/10)^2.17 * L^2, i.e. 1 : 18 : 98 for
# L = 10 : 20 : 30, so the width scales roughly inversely.
DEFAULT_WIDTH = {10: 0.40, 16: 0.20, 20: 0.20, 24: 0.10, 30: 0.10, 32: 0.10}


def auto_width(L, eps_d_step):
    w = DEFAULT_WIDTH.get(int(L))
    if w is None:
        w = max(4.0 * eps_d_step,
                0.40 * (10.0 / float(L)) ** 2.17 * (10.0 / float(L)) ** 2)
    # snap to a whole number of steps
    n = max(int(round(w / eps_d_step)), 1)
    return n * eps_d_step


def load_anchors(system, trace_dir):
    path = os.path.join(trace_dir, TRACE[system])
    if not os.path.exists(path):
        raise SystemExit(f"trace log not found: {path}")
    recs = clean(parse_trace(path))
    if not recs:
        raise SystemExit(f"no usable criticality records in {path}")
    e = np.array([r["eps_d"] for r in recs], float)
    return recs, e


def anchor_at(recs, e, eps_d, s_window=0.06):
    """Logged L=10 (eps_nd, mu, s) at eps_d, linearly interpolated.

    s is taken as the local median rather than a single step's value: it is
    refitted from a finite sample at every step and scatters accordingly.
    Refuses to extrapolate -- a segment whose left edge is outside the logged
    range is reported so the caller can drop it.
    """
    if eps_d < e.min() - 1e-9 or eps_d > e.max() + 1e-9:
        return None
    eps_nd = float(np.interp(eps_d, e, [r["eps_nd"] for r in recs]))
    mu = float(np.interp(eps_d, e, [r["mu"] for r in recs]))
    m = np.abs(e - eps_d) <= s_window
    w = float(s_window)
    while m.sum() < 3 and w < (e.max() - e.min()):
        w *= 2.0
        m = np.abs(e - eps_d) <= w
    s = float(np.median([recs[j]["s"] for j in np.flatnonzero(m)])) \
        if m.any() else float("nan")
    gap = float(np.min(np.abs(e - eps_d)))
    return dict(eps_nd=eps_nd, mu=mu, s=s, gap=gap, n_local=int(m.sum()))


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--system", default="l", choices=sorted(TRACE))
    p.add_argument("--L", type=int, default=10)
    p.add_argument("--eps-d-min", type=float, default=0.0)
    p.add_argument("--eps-d-max", type=float, default=6.0)
    p.add_argument("--eps-d-step", type=float, default=0.02)
    p.add_argument("--width", default="auto",
                   help="segment width in eps_d, or 'auto' (per-L default)")
    p.add_argument("--overlap", type=float, default=0.0,
                   help="extend each segment by this much past its right edge "
                        "so neighbours overlap; the overlap is the "
                        "segment-to-segment consistency check")
    p.add_argument("--seeds", default="1",
                   help="space-separated seeds, e.g. \"1 2 3\"")
    p.add_argument("--trace-dir", default="../results_logs")
    p.add_argument("--fix-s", action="store_true",
                   help="pass the logged s to the sweep instead of refitting "
                        "it at every step")
    p.add_argument("--list-anchors", action="store_true",
                   help="print the anchors and the logged coverage, no manifest")
    p.add_argument("--out", default=None, help="write here instead of stdout")
    args = p.parse_args(argv)

    recs, e = load_anchors(args.system, args.trace_dir)
    step = float(args.eps_d_step)
    width = auto_width(args.L, step) if args.width == "auto" \
        else float(args.width)
    n_per_seg = max(int(round(width / step)), 1)
    seeds = [int(x) for x in str(args.seeds).split()]

    if args.list_anchors:
        print(f"system {args.system}: {len(recs)} logged criticality records, "
              f"eps_d {e.min():.3f} .. {e.max():.3f}")
        print(f"segment width at L={args.L}: {width:g} "
              f"({n_per_seg} steps of {step:g})")
        print(f"{'eps_d':>8} {'eps_nd':>10} {'mu':>10} {'s':>8} {'gap':>7}")

    lines = []
    seg = 0
    x = float(args.eps_d_min)
    eps_max = float(args.eps_d_max)
    while x < eps_max - 1e-9:
        lo = x
        hi = min(lo + width + args.overlap, eps_max)
        # snap hi to the step grid measured from lo
        n = max(int(round((hi - lo) / step)), 1)
        hi = lo + n * step

        if abs(lo) < 1e-12:
            a = dict(eps_nd=None, mu=None, s=None, gap=0.0, n_local=0)
            tok = ("-", "-", "-")
        else:
            a = anchor_at(recs, e, lo)
            if a is None:
                print(f"# skip segment {seg}: eps_d={lo:.4f} is outside the "
                      f"logged range [{e.min():.3f},{e.max():.3f}]",
                      file=sys.stderr)
                seg += 1
                x = lo + width
                continue
            tok = (f"{a['eps_nd']:.6f}", f"{a['mu']:.6f}",
                   (f"{a['s']:.6f}" if args.fix_s else "-"))

        if args.list_anchors:
            en = "-" if a["eps_nd"] is None else f"{a['eps_nd']:10.5f}"
            mu = "-" if a["mu"] is None else f"{a['mu']:10.5f}"
            sv = "-" if a["s"] is None else f"{a['s']:8.4f}"
            print(f"{lo:>8.3f} {en:>10} {mu:>10} {sv:>8} {a['gap']:>7.3f}")

        for sd in seeds:
            lines.append(f"{args.system} {args.L} {seg:03d} "
                         f"{lo:.6f} {hi:.6f} {tok[0]} {tok[1]} {tok[2]} {sd}")
        seg += 1
        x = lo + width

    if args.list_anchors:
        return 0

    text = "\n".join(lines) + ("\n" if lines else "")
    if args.out:
        with open(args.out, "w") as fh:
            fh.write(text)
        print(f"{len(lines)} tasks -> {args.out}", file=sys.stderr)
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
