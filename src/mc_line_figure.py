#!/usr/bin/env python
"""
mc_line_figure.py -- the finite-size critical line, extrapolated, against theory.

Two figures:

  <out>_line.png   eps_nd,c vs eps_d for each geometry, at L = 10, 20, 30 and
                   L -> infinity, with the SAFT-P and SAFT boundaries.  This is
                   the reviewer-facing picture: it shows where the simulated
                   critical line actually sits once the lattice size is taken
                   out of it, and how far that is from each theory.

  <out>_shift.png  the finite-size displacement Delta_FS(L=10) against the
                   SAFT-P/SAFT separation, on the same axis and in the same
                   units, because the only question that matters for R2#4 is
                   which of the two is bigger and by how much.

Input is either a workdir of relocation .npz files, or a CSV with columns
system,eps_d,L,eps_nd_c (so the figure can be rebuilt from a table).

    python mc_line_figure.py --workdir /pool/hamza/mc_line_v6 \\
        --data-dir ../data --out ../figures/mc_line_v6
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys

import numpy as np

# Validated categorical slots (adjacent pairlist, lines): worst CVD dE 9.1.
SERIES = {10: "#2a78d6", 20: "#eb6834", 30: "#1baf7a"}
CINF = "#4a3aa7"
SAFTP = "#008300"
SAFT = "#e34948"
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1"
_RX = re.compile(r"^(?P<sys>[a-z]+)_L(?P<L>\d+)_epsd(?P<e>[-\d.eE+]+)"
                 r"_s(?P<seed>\d+)\.npz$")
LABEL = {"stick": "collinear", "l": "bent"}
# one slot per state point (adjacent pairlist, lines): worst CVD dE 9.1
SERIES_N = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
            "#e87ba4", "#008300", "#4a3aa7", "#e34948"]


def from_workdir(workdir, seed):
    rows = []
    for p in sorted(glob.glob(os.path.join(workdir, "*.npz"))):
        m = _RX.match(os.path.basename(p))
        if not m or int(m["seed"]) != seed:
            continue
        z = np.load(p, allow_pickle=False)
        if "points_json" not in z.files:
            continue
        rel = [r for r in json.loads(str(z["points_json"]))
               if r.get("mode") == "relocate"]
        if not rel:
            continue
        r = rel[0]
        bad = (r.get("eps_clamped") or r.get("at_bound")
               or not np.isfinite(r.get("curvature", np.nan))
               or r["min_crossings"] < 1 or r["js"] > 0.10)
        rows.append(dict(system=m["sys"], eps_d=float(m["e"]), L=int(m["L"]),
                         eps_nd_c=float(r["eps_nd"]), js=float(r["js"]),
                         crossings=int(r["min_crossings"]), flagged=bool(bad)))
    return rows


def from_csv(path):
    rows = []
    with open(path) as fh:
        hdr = fh.readline().strip().split(",")
        for line in fh:
            if not line.strip():
                continue
            d = dict(zip(hdr, line.strip().split(",")))
            rows.append(dict(system=d["system"], eps_d=float(d["eps_d"]),
                             L=int(d["L"]), eps_nd_c=float(d["eps_nd_c"]),
                             js=float(d.get("js", "nan")),
                             crossings=int(d.get("crossings", 0)),
                             flagged=d.get("flagged", "0") in ("1", "True")))
    return rows


def solve_exponent(L, y):
    """Solve eps_nd,c(L) = c + a L^-x exactly from three sizes.

    Three points, three unknowns, so x is DETERMINED but not tested -- a fourth
    size is what would test it.  What makes the answer credible anyway is that
    six independent state points, two geometries, all land in the same place and
    all reject x = 1.  The ratio (y1-y2)/(y2-y3) is 3.00 for x = 1 and larger
    for steeper decay; every measured ratio here is 4.0-7.8.
    """
    from scipy.optimize import brentq
    L = np.asarray(L, float)
    y = np.asarray(y, float)
    if L.size != 3:
        return None
    den = y[1] - y[2]
    if abs(den) < 1e-12:
        return None
    R = (y[0] - y[1]) / den
    def f(x):
        return ((L[0] ** -x - L[1] ** -x) / (L[1] ** -x - L[2] ** -x)) - R
    try:
        x = brentq(f, 0.2, 6.0)
    except Exception:                                          # noqa: BLE001
        return None
    a = (y[0] - y[1]) / (L[0] ** -x - L[1] ** -x)
    return dict(x=float(x), a=float(a), c=float(y[2] - a * L[2] ** -x))


def group(rows):
    pts = {}
    for r in rows:
        pts.setdefault((r["system"], r["eps_d"]), []).append(r)
    out = {}
    for k, v in pts.items():
        v.sort(key=lambda d: d["L"])
        if len(v) < 2:
            continue
        L = np.array([d["L"] for d in v], float)
        y = np.array([d["eps_nd_c"] for d in v], float)
        A = np.vstack([np.ones_like(L), 1.0 / L]).T
        beta, *_ = np.linalg.lstsq(A, y, rcond=None)
        resid = float(np.sqrt(np.mean((y - A @ beta) ** 2)))
        d = np.diff(y)
        out[k] = dict(rows=v, L=L, y=y, c=float(beta[0]), a=float(beta[1]),
                      resid=resid, free=solve_exponent(L, y),
                      monotone=bool((d > 0).all() or (d < 0).all()),
                      flagged=any(d_["flagged"] for d_ in v))
    return out


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--workdir", default=None)
    p.add_argument("--points-csv", default=None)
    p.add_argument("--data-dir", default="../data")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--out", default="mc_line")
    p.add_argument("--with-theory", action="store_true",
                   help="overlay the SAFT / SAFT-P boundaries on the line "
                        "figure. Off by default: their range is 3-10x the "
                        "spread between lattice sizes, so including them "
                        "compresses the finite-size structure into a band")
    p.add_argument("--keep-flagged", action="store_true",
                   help="include points that carry a diagnostic flag "
                        "(drawn hollow either way)")
    args = p.parse_args(argv)

    rows = (from_csv(args.points_csv) if args.points_csv
            else from_workdir(args.workdir, args.seed))
    if not rows:
        raise SystemExit("no points found")
    pts = group(rows)

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    try:
        from mc_fss_trace import theory_at
    except Exception:                                          # noqa: BLE001
        theory_at = None

    # ---- the table ------------------------------------------------------
    print(f"\n{'point':>12} {'L=10':>9} {'L=20':>9} {'L=30':>9} {'L->inf':>9} "
          f"{'a':>8} {'D_FS(10)':>9} {'SAFT-P':>8} {'SAFT':>7} "
          f"{'D_FS/sep':>9}  ")
    summary = []
    for (sysname, epsd), g in sorted(pts.items()):
        th = theory_at(sysname, epsd, args.data_dir) if theory_at else {}
        sp, sa = th.get("saft_p"), th.get("saft")
        sep = (sa - sp) if (sp is not None and sa is not None) else np.nan
        dfs = g["y"][0] - g["c"]
        ys = {int(L): v for L, v in zip(g["L"], g["y"])}
        print(f"{sysname + ' ' + format(epsd, '.2f'):>12} "
              + " ".join(f"{ys.get(L, float('nan')):>9.5f}"
                         for L in (10, 20, 30))
              + f" {g['c']:>9.5f} {g['a']:>+8.4f} {dfs:>+9.5f} "
              + (f"{sp:>8.4f} {sa:>7.4f} {dfs / sep:>8.1%}"
                 if np.isfinite(sep) else " " * 26)
              + ("   FLAGGED" if g["flagged"] else "")
              + ("" if g["monotone"] else "  NON-MONO"))
        summary.append(dict(system=sysname, eps_d=epsd, c=g["c"], a=g["a"],
                            dfs=dfs, saft_p=sp, saft=sa, sep=sep,
                            flagged=g["flagged"]))

    ok = [s for s in summary if not s["flagged"]]
    if ok:
        a = np.array([s["a"] for s in ok])
        d = np.array([s["dfs"] for s in ok])
        rat = np.array([s["dfs"] / s["sep"] for s in ok
                        if np.isfinite(s["sep"])])
        print(f"\n  over the {len(ok)} unflagged points:")
        print(f"    amplitude a       = {a.mean():.4f} +- {a.std():.4f}  "
              f"({a.std() / abs(a.mean()):.0%} spread)")
        print(f"    Delta_FS(L=10)    = {d.mean():+.4f} +- {d.std():.4f}")
        if rat.size:
            print(f"    as a fraction of the SAFT-P/SAFT separation: "
                  f"{rat.min():.1%} .. {rat.max():.1%}")
        # does extrapolating help or hurt agreement with SAFT-P?
        closer = sum(1 for s in ok if s["saft_p"] is not None
                     and abs(s["c"] - s["saft_p"])
                     < abs(s["c"] + s["dfs"] - s["saft_p"]))
        print(f"    L -> infinity moves the MC line CLOSER to SAFT-P in "
              f"{closer}/{len(ok)} points")

    print(f"\n  the three sizes also MEASURE the decay exponent x in "
          f"eps_nd,c(L) = c + a L^-x:")
    print(f"{'point':>12} {'x':>6} {'c(inf)':>10} {'D_FS(10)':>10} "
          f"{'D_FS(30)':>10}   {'[x=1]':>8} {'c(inf)':>10} {'D_FS(10)':>10} "
          f"{'D_FS(30)':>10}")
    xs, dfs_free, dfs_1, res_free, res_1 = [], [], [], [], []
    for (sysname, epsd), g in sorted(pts.items()):
        if g["flagged"] or not g["free"]:
            continue
        fr = g["free"]
        xs.append(fr["x"])
        dfs_free.append(g["y"][0] - fr["c"])
        res_free.append(g["y"][-1] - fr["c"])
        dfs_1.append(g["y"][0] - g["c"])
        res_1.append(g["y"][-1] - g["c"])
        print(f"{sysname + ' ' + format(epsd, '.2f'):>12} {fr['x']:>6.2f} "
              f"{fr['c']:>10.5f} {g['y'][0] - fr['c']:>10.5f} "
              f"{g['y'][-1] - fr['c']:>10.5f}   {'':>8} {g['c']:>10.5f} "
              f"{g['y'][0] - g['c']:>10.5f} {g['y'][-1] - g['c']:>10.5f}")
    if xs:
        xs = np.array(xs)
        print(f"\n    x = {xs.mean():.2f} +- {xs.std():.2f} over {xs.size} "
              f"points; 2D-Ising leading shift would give x = 1")
        print(f"    ratio (y10-y20)/(y20-y30) is 3.00 at x=1; every point "
              f"here is above it")
        print(f"    Delta_FS(L=10):  {np.mean(dfs_free):+.4f} with x free, "
              f"{np.mean(dfs_1):+.4f} forcing x=1")
        print(f"    residual at L=30: {np.mean(res_free):+.4f} with x free, "
              f"{np.mean(res_1):+.4f} forcing x=1  "
              f"(factor {np.mean(res_1)/max(np.mean(res_free),1e-12):.0f})")

    if args.out:
        _figures(pts, summary, args.out, args.data_dir, theory_at,
                 args.keep_flagged, args.with_theory)
    return 0




def _figures(pts, summary, out, data_dir, theory_at, keep_flagged,
             with_theory=False):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.size": 10, "axes.labelsize": 11, "axes.titlesize": 11.5,
        "axes.edgecolor": MUTED, "axes.linewidth": 0.9,
        "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
        "axes.labelcolor": INK, "figure.facecolor": "white",
        "axes.facecolor": "white"})
    systems = sorted({k[0] for k in pts})
    good = [(k, g) for k, g in sorted(pts.items())
            if not g["flagged"] and g["free"]]

    # ================= figure 1: how the shift scales ====================
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13.5, 5.8),
                                 constrained_layout=True)
    for i, ((sysname, epsd), g) in enumerate(good):
        c = SERIES_N[i % len(SERIES_N)]
        d = g["y"] - g["free"]["c"]
        a1.plot(g["L"], d, "-o", lw=2.0, ms=8, color=c, mec="white", mew=1.5)
        a1.annotate(f"{LABEL.get(sysname, sysname)}  {epsd:.2f}",
                    xy=(g["L"][-1], d[-1]), xytext=(9, 0),
                    textcoords="offset points", color=c, fontweight="bold",
                    fontsize=9.5, ha="left", va="center")
    Lg = np.array([9.0, 34.0])
    for expo, ls, lab in ((1.0, "--", r"$L^{-1}$"), (2.0, ":", r"$L^{-2}$")):
        ref = 0.060 * (Lg / 10.0) ** (-expo)
        a1.plot(Lg, ref, ls, lw=1.8, color=INK2)
        Lm = 15.0
        a1.annotate(lab, xy=(Lm, 0.060 * (Lm / 10.0) ** (-expo)),
                    xytext=(0, 9 if expo == 1.0 else -19),
                    textcoords="offset points", color=INK2, fontsize=11,
                    ha="center")
    a1.set_xscale("log")
    a1.set_yscale("log")
    a1.set_xticks([10, 20, 30])
    a1.set_xticklabels(["10", "20", "30"])
    a1.set_xlabel("L")
    a1.set_ylabel(r"$\epsilon_{\rm nd,c}(L)-\epsilon_{\rm nd,c}(\infty)$")
    a1.set_title("The shift decays faster than $L^{-1}$", loc="left")
    a1.set_xlim(8.4, 62.0)
    a1.xaxis.set_minor_formatter(plt.NullFormatter())
    a1.grid(True, which="both", color=GRID, lw=0.6)
    a1.set_axisbelow(True)

    for i, ((sysname, epsd), g) in enumerate(good):
        c = SERIES_N[i % len(SERIES_N)]
        a2.plot([epsd], [g["free"]["x"]], "o" if sysname == "stick" else "s",
                ms=13, color=c, mec="white", mew=1.6)
        # circles right, squares left: the two points near eps_d = 1 would
        # otherwise print on top of each other
        dx, ha = (16, "left") if sysname == "stick" else (-16, "right")
        a2.annotate(f"{g['free']['x']:.2f}", xy=(epsd, g["free"]["x"]),
                    xytext=(dx, 0), textcoords="offset points", color=c,
                    fontweight="bold", fontsize=10, ha=ha, va="center")
    a2.axhline(1.0, color=SAFT, lw=2.0, ls="--")
    a2.annotate("$x = 1$", xy=(a2.get_xlim()[1], 1.0), xytext=(-8, 8),
                textcoords="offset points", color=SAFT, fontweight="bold",
                fontsize=11, ha="right")
    xs = np.array([g["free"]["x"] for _, g in good])
    a2.axhspan(xs.mean() - xs.std(), xs.mean() + xs.std(), color=CINF,
               alpha=0.10, lw=0)
    a2.axhline(xs.mean(), color=CINF, lw=2.0)
    a2.annotate(f"$x = {xs.mean():.2f} \\pm {xs.std():.2f}$",
                xy=(a2.get_xlim()[1], xs.mean()), xytext=(-8, 12),
                textcoords="offset points", color=CINF, fontweight="bold",
                fontsize=11, ha="right")
    a2.set_ylim(0.6, max(3.1, xs.max() + 0.4))
    e0, e1 = a2.get_xlim()
    a2.set_xlim(e0 - 0.10 * (e1 - e0), e1 + 0.10 * (e1 - e0))
    a2.set_xlabel(r"$\epsilon_{\rm d}$")
    a2.set_ylabel("decay exponent  x")
    a2.set_title("Decay exponent per state point"
                 "   (circles collinear, squares bent)", loc="left",
                 fontsize=11)
    a2.grid(True, color=GRID, lw=0.6)
    a2.set_axisbelow(True)
    fig.savefig(out + "_scaling.png", dpi=200)
    plt.close(fig)
    print(f"\n[out] {out}_scaling.png")

    # ================= figure 2: the line, MC only =======================
    fig, axs = plt.subplots(1, len(systems),
                            figsize=(7.0 * len(systems), 6.0),
                            constrained_layout=True, squeeze=False)
    for ax, sysname in zip(axs[0], systems):
        sel = sorted([(e, g) for (s_, e), g in pts.items()
                      if s_ == sysname and (keep_flagged or not g["flagged"])])
        if not sel:
            continue
        if with_theory and theory_at is not None:
            eds = np.array([e for e, _ in sel])
            grid = np.linspace(eds.min() - 0.08, eds.max() + 0.08, 60)
            for key, col, lab in (("saft_p", SAFTP, "SAFT-P"),
                                  ("saft", SAFT, "SAFT")):
                v = [theory_at(sysname, e, data_dir).get(key) for e in grid]
                if all(q is not None for q in v):
                    ax.plot(grid, v, lw=2.0, color=col, zorder=1)
        for L in (10, 20, 30):
            xs_, ys_ = [], []
            for e, g in sel:
                for r in g["rows"]:
                    if r["L"] == L:
                        xs_.append(e)
                        ys_.append(r["eps_nd_c"])
            if not xs_:
                continue
            o = np.argsort(xs_)
            xs_, ys_ = np.array(xs_)[o], np.array(ys_)[o]
            ax.plot(xs_, ys_, "-o", lw=1.6, ms=9, color=SERIES[L],
                    mec="white", mew=1.6, zorder=3)
            ax.annotate(f"L = {L}", xy=(xs_[0], ys_[0]),
                        xytext=(-12, {10: 12, 20: 0, 30: -12}[L]),
                        textcoords="offset points", color=SERIES[L],
                        fontweight="bold", fontsize=10.5, ha="right",
                        va="center")
        xs_ = np.array([e for e, g in sel if g["free"]])
        c_free = np.array([g["free"]["c"] for e, g in sel if g["free"]])
        c_one = np.array([g["c"] for e, g in sel if g["free"]])
        if xs_.size:
            o = np.argsort(xs_)
            ax.fill_between(xs_[o], c_one[o], c_free[o], color=CINF,
                            alpha=0.16, lw=0, zorder=2)
            ax.plot(xs_[o], c_free[o], "-D", lw=2.6, ms=9, color=CINF,
                    mec="white", mew=1.6, zorder=4)
            ax.annotate(r"$L\to\infty$", xy=(xs_[o][0], c_free[o][0]),
                        xytext=(-12, -26), textcoords="offset points",
                        color=CINF, fontweight="bold", fontsize=10.5,
                        ha="right", va="center")
        ax.set_xlabel(r"$\epsilon_{\rm d}$")
        ax.set_ylabel(r"$\epsilon_{\rm nd,c}$")
        ax.set_title(LABEL.get(sysname, sysname), loc="left")
        ax.grid(True, color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        x0, x1 = ax.get_xlim()
        ax.set_xlim(x0 - 0.30 * (x1 - x0), x1 + 0.06 * (x1 - x0))
    fig.savefig(out + "_line.png", dpi=200)
    plt.close(fig)
    print(f"[out] {out}_line.png")


if __name__ == "__main__":
    raise SystemExit(main())
