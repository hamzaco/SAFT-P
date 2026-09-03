#!/usr/bin/env python
"""
mc_line_merge.py -- collect the segment .npz files, stitch each lattice size
into one critical line eps_nd,c(eps_d; L), and extrapolate to L -> infinity.

The finite-size analysis
------------------------
For a critical point in the 2D Ising universality class the leading finite-size
shift of a non-universal critical coupling is

    eps_nd,c(L) = eps_nd,c(inf) + a * L^(-1/nu) + ...      nu = 1

so at each eps_d we fit that two-parameter form to the available sizes.  With
three sizes (L = 10, 20, 30) that leaves one degree of freedom, and the
residual is reported: it is the only internal check that the leading-order form
is adequate over this range of L, and it should be quoted rather than hidden.

`--nu-free` fits the exponent too, which needs four or more sizes to mean
anything; it is there to test the assumption, not to produce the headline
number.

What comes out
--------------
  <prefix>_points.csv     every segment step, all diagnostics, flagged
  <prefix>_lines.csv      stitched eps_nd,c(eps_d) per L on the common grid
  <prefix>_extrap.csv     eps_nd,c(inf)(eps_d) with the fit residual and slope
  <prefix>_lines.png      the three lines + the extrapolation
  <prefix>_shift.png      Delta_FS(eps_d) = eps_nd,c(L) - eps_nd,c(inf)
  <prefix>_diag.png       ESS/n, min crossings and JS along the sweep

Usage
-----
    python mc_line_merge.py --workdir /pool/hamza/mc_line --outdir ../figures
    python mc_line_merge.py --workdir ... --system l --sizes 10,20,30 \
        --drop-nonergodic --prefix ../figures/fig_fss_line_l
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np


# ----------------------------------------------------------------------------
# loading
# ----------------------------------------------------------------------------
def load_segments(workdir, system=None, sizes=None, seeds=None):
    rows = []
    for path in sorted(glob.glob(os.path.join(workdir, "*.npz"))):
        base = os.path.basename(path)
        if base.startswith("ising_ref"):
            continue
        try:
            z = np.load(path, allow_pickle=False)
        except Exception as exc:                               # noqa: BLE001
            print(f"[skip] {base}: {exc}", file=sys.stderr)
            continue
        if "points_json" not in z.files:
            continue
        sysname = str(z["system"])
        L = int(z["L"])
        seed = int(z["seed"])
        if system and sysname != system:
            continue
        if sizes and L not in sizes:
            continue
        if seeds and seed not in seeds:
            continue
        pts = json.loads(str(z["points_json"]))
        for r in pts:
            r = dict(r)
            r["system"] = sysname
            r["L"] = L
            r["seed"] = seed
            r["segment"] = base
            r["eps_d_seg_start"] = float(z["eps_d_start"])
            rows.append(r)
        if not bool(z["complete"]):
            print(f"[warn] {base}: segment incomplete "
                  f"({len(pts)} steps, ends at eps_d="
                  f"{pts[-1]['eps_d'] if pts else float('nan'):.4f})",
                  file=sys.stderr)
    return rows


def flag_rows(rows, min_crossings=1, ess_floor=0.05):
    """Mark rows that should not carry weight, and say why."""
    for r in rows:
        why = []
        if r.get("min_crossings", 0) < min_crossings:
            why.append("no-tunnelling")
        if r.get("ess_frac", 1.0) < ess_floor:
            why.append("low-ESS")
        if r.get("stalled") and not r.get("rescued_by_scan"):
            why.append("stalled")
        if r.get("at_bound"):
            why.append("at-bound")
        # A relocation whose JS profile has no interior parabola walked to the
        # edge of its window: the anchor is not close enough, or there is not
        # enough statistics to resolve a minimum.  Widening the window is the
        # wrong fix -- it just lets it walk further.
        if r.get("mode") == "relocate" and not np.isfinite(
                r.get("curvature", np.nan)):
            why.append("no-minimum")
        r["flags"] = ",".join(why)
        r["ok"] = (len(why) == 0)
    return rows


# ----------------------------------------------------------------------------
# stitching
# ----------------------------------------------------------------------------
def stitch(rows, L, grid, seeds_together="mean", drop_flagged=False):
    """One eps_nd,c(eps_d) curve per L on a common eps_d grid.

    Segments overlap by design (submit_mc_line.sh sets OVERLAP), and seeds are
    independent repeats of the same segment.  Both are averaged at each grid
    point; the spread across whatever landed in a bin is the error bar, so an
    overlap disagreement shows up as a large bar rather than being averaged
    away silently.
    """
    sel = [r for r in rows if r["L"] == L and (r["ok"] or not drop_flagged)]
    if not sel:
        return None
    e = np.array([r["eps_d"] for r in sel], float)
    y = np.array([r["eps_nd"] for r in sel], float)
    m = np.array([r["mu"] for r in sel], float)

    uniq = np.unique(np.round(e, 6))
    d = np.diff(uniq)
    half = 0.5 * (float(np.median(d)) if d.size and np.isfinite(np.median(d))
                  else float(np.median(np.diff(grid))) if grid.size > 1
                  else 0.01)
    out_y = np.full(grid.size, np.nan)
    out_s = np.full(grid.size, np.nan)
    out_m = np.full(grid.size, np.nan)
    out_n = np.zeros(grid.size, int)
    for i, g in enumerate(grid):
        k = np.abs(e - g) <= half
        if not k.any():
            continue
        out_y[i] = float(np.mean(y[k]))
        out_m[i] = float(np.mean(m[k]))
        out_n[i] = int(k.sum())
        out_s[i] = float(np.std(y[k], ddof=1)) if k.sum() > 1 else np.nan
    return dict(eps_d=grid, eps_nd=out_y, eps_nd_sd=out_s, mu=out_m, n=out_n)


# ----------------------------------------------------------------------------
# finite-size extrapolation
# ----------------------------------------------------------------------------
def extrapolate(curves, nu=1.0, nu_free=False):
    """eps_nd,c(L) = eps_nd,c(inf) + a L^(-1/nu), fitted at each eps_d.

    Returns arrays over the common grid.  `resid` is the RMS residual of the
    fit -- with three sizes and two parameters this is one degree of freedom,
    so it is a real, if weak, test of the leading-order form.  Points where
    fewer than two sizes are available come back as nan.
    """
    Ls = np.array(sorted(curves), float)
    grid = curves[int(Ls[0])]["eps_d"]
    Y = np.vstack([curves[int(L)]["eps_nd"] for L in Ls])       # (nL, ngrid)

    inf = np.full(grid.size, np.nan)
    slope = np.full(grid.size, np.nan)
    resid = np.full(grid.size, np.nan)
    nu_fit = np.full(grid.size, np.nan)
    nsz = np.zeros(grid.size, int)

    for j in range(grid.size):
        col = Y[:, j]
        k = np.isfinite(col)
        nsz[j] = int(k.sum())
        if k.sum() < 2:
            continue
        if nu_free and k.sum() >= 4:
            from scipy.optimize import curve_fit
            try:
                popt, _ = curve_fit(
                    lambda L, c, a, inv_nu: c + a * L ** (-inv_nu),
                    Ls[k], col[k], p0=[col[k][-1], col[k][0] - col[k][-1], 1.0],
                    maxfev=20000)
                inf[j], slope[j], nu_fit[j] = popt[0], popt[1], 1.0 / popt[2]
                pred = popt[0] + popt[1] * Ls[k] ** (-popt[2])
                resid[j] = float(np.sqrt(np.mean((col[k] - pred) ** 2)))
                continue
            except Exception:                                  # noqa: BLE001
                pass
        x = Ls[k] ** (-1.0 / nu)
        A = np.vstack([np.ones_like(x), x]).T
        beta, *_ = np.linalg.lstsq(A, col[k], rcond=None)
        inf[j], slope[j] = float(beta[0]), float(beta[1])
        pred = A @ beta
        resid[j] = float(np.sqrt(np.mean((col[k] - pred) ** 2)))
        nu_fit[j] = nu

    return dict(eps_d=grid, eps_nd_inf=inf, slope=slope, resid=resid,
                nu=nu_fit, n_sizes=nsz, Ls=Ls)


# ----------------------------------------------------------------------------
# output
# ----------------------------------------------------------------------------
def write_csv(path, header, cols):
    n = len(cols[0])
    with open(path, "w") as fh:
        fh.write(",".join(header) + "\n")
        for i in range(n):
            fh.write(",".join(
                ("" if (isinstance(c[i], float) and not np.isfinite(c[i]))
                 else (f"{c[i]:.8g}" if isinstance(c[i], (float, np.floating))
                       else str(c[i])))
                for c in cols) + "\n")
    print(f"[out] {path}")


def make_figures(curves, ext, prefix, system, rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    Ls = sorted(curves)

    # --- 1. the lines ----------------------------------------------------
    fig, ax = plt.subplots(figsize=(6.2, 4.4), constrained_layout=True)
    for L in Ls:
        c = curves[L]
        k = np.isfinite(c["eps_nd"])
        sd = np.where(np.isfinite(c["eps_nd_sd"]), c["eps_nd_sd"], 0.0)
        ax.errorbar(c["eps_d"][k], c["eps_nd"][k], yerr=sd[k],
                    marker="o", ms=2.5, lw=1.0, capsize=0, label=f"L = {L}")
    k = np.isfinite(ext["eps_nd_inf"])
    ax.plot(ext["eps_d"][k], ext["eps_nd_inf"][k], "k--", lw=1.6,
            label=r"$L\to\infty$")
    ax.set_xlabel(r"$\epsilon_{\rm d}$")
    ax.set_ylabel(r"$\epsilon_{\rm nd,c}$")
    ax.set_title(f"critical line, {system}")
    ax.legend(frameon=False)
    fig.savefig(prefix + "_lines.png", dpi=200)
    plt.close(fig)
    print(f"[out] {prefix}_lines.png")

    # --- 2. the finite-size shift ----------------------------------------
    fig, ax = plt.subplots(figsize=(6.2, 4.0), constrained_layout=True)
    for L in Ls:
        c = curves[L]
        d = c["eps_nd"] - ext["eps_nd_inf"]
        k = np.isfinite(d)
        ax.plot(c["eps_d"][k], d[k], marker="o", ms=2.5, lw=1.0,
                label=f"L = {L}")
    ax.axhline(0.0, color="k", lw=0.8)
    ax.set_xlabel(r"$\epsilon_{\rm d}$")
    ax.set_ylabel(r"$\epsilon_{\rm nd,c}(L) - \epsilon_{\rm nd,c}(\infty)$")
    ax.set_title(r"finite-size displacement $\Delta_{\rm FS}$")
    ax.legend(frameon=False)
    fig.savefig(prefix + "_shift.png", dpi=200)
    plt.close(fig)
    print(f"[out] {prefix}_shift.png")

    # --- 3. diagnostics along the sweep ----------------------------------
    fig, axs = plt.subplots(3, 1, figsize=(6.4, 7.0), sharex=True,
                            constrained_layout=True)
    for L in Ls:
        sel = [r for r in rows if r["L"] == L]
        if not sel:
            continue
        e = np.array([r["eps_d"] for r in sel])
        o = np.argsort(e)
        e = e[o]
        axs[0].plot(e, np.array([r["ess_frac"] for r in sel])[o], ".",
                    ms=3, label=f"L = {L}")
        axs[1].plot(e, np.array([r["min_crossings"] for r in sel])[o], ".",
                    ms=3)
        axs[2].plot(e, np.array([r["js"] for r in sel])[o], ".", ms=3)
    axs[0].axhline(0.05, color="r", lw=0.8, ls=":")
    axs[0].set_ylabel("ESS / n")
    axs[0].legend(frameon=False, ncol=3)
    axs[1].axhline(1, color="r", lw=0.8, ls=":")
    axs[1].set_yscale("symlog", linthresh=1)
    axs[1].set_ylabel("min replica crossings")
    axs[2].set_ylabel("JS at the minimum")
    axs[2].set_xlabel(r"$\epsilon_{\rm d}$")
    fig.savefig(prefix + "_diag.png", dpi=200)
    plt.close(fig)
    print(f"[out] {prefix}_diag.png")


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--workdir", default="/pool/hamza/mc_line")
    p.add_argument("--outdir", default="../figures")
    p.add_argument("--prefix", default=None,
                   help="output path prefix (default: <outdir>/mc_line_<system>)")
    p.add_argument("--system", default="l")
    p.add_argument("--sizes", default=None,
                   help="comma list, e.g. 10,20,30 (default: whatever is there)")
    p.add_argument("--seeds", default=None, help="comma list")
    p.add_argument("--eps-d-step", type=float, default=0.02)
    p.add_argument("--nu", type=float, default=1.0,
                   help="correlation-length exponent; 1 for 2D Ising")
    p.add_argument("--nu-free", action="store_true",
                   help="fit nu as well (needs >=4 sizes to mean anything)")
    p.add_argument("--drop-nonergodic", action="store_true",
                   help="exclude steps where a replica never tunnelled, or "
                        "ESS/n < 0.05, or L-BFGS-B stalled unrescued")
    p.add_argument("--min-crossings", type=int, default=1)
    p.add_argument("--ess-floor", type=float, default=0.05)
    p.add_argument("--no-figures", action="store_true")
    args = p.parse_args(argv)

    sizes = [int(x) for x in args.sizes.split(",")] if args.sizes else None
    seeds = [int(x) for x in args.seeds.split(",")] if args.seeds else None

    rows = load_segments(args.workdir, args.system, sizes, seeds)
    if not rows:
        raise SystemExit(f"no segment results found in {args.workdir}")
    rows = flag_rows(rows, args.min_crossings, args.ess_floor)

    Ls = sorted({r["L"] for r in rows})
    print(f"loaded {len(rows)} steps, sizes {Ls}, "
          f"seeds {sorted({r['seed'] for r in rows})}")
    bad = [r for r in rows if not r["ok"]]
    if bad:
        from collections import Counter
        print(f"flagged {len(bad)}/{len(rows)} steps: "
              f"{dict(Counter(r['flags'] for r in bad))}")
        if not args.drop_nonergodic:
            print("  (kept -- pass --drop-nonergodic to exclude them)")

    os.makedirs(args.outdir, exist_ok=True)
    prefix = args.prefix or os.path.join(args.outdir, f"mc_line_{args.system}")

    # --- every step, for the record --------------------------------------
    keys = ["system", "L", "seed", "segment", "mode", "eps_d", "d_eps_d",
            "eps_nd", "mu", "s", "r", "js", "curvature", "ess_frac",
            "mean_rho", "min_crossings", "total_crossings", "mean_accept",
            "rho_spread", "stalled", "rescued_by_scan", "at_bound",
            "opt_success", "wall_seconds", "flags", "ok"]
    rows_sorted = sorted(rows, key=lambda r: (r["L"], r["seed"], r["eps_d"]))
    write_csv(prefix + "_points.csv", keys,
              [[r.get(k) for r in rows_sorted] for k in keys])

    # --- common grid ------------------------------------------------------
    e_all = np.array([r["eps_d"] for r in rows], float)
    step = float(args.eps_d_step)
    g0 = step * np.floor(e_all.min() / step + 0.5)
    g1 = step * np.ceil(e_all.max() / step - 0.5)
    grid = np.round(np.arange(g0, g1 + 0.5 * step, step), 10)

    curves = {}
    for L in Ls:
        c = stitch(rows, L, grid, drop_flagged=args.drop_nonergodic)
        if c is not None:
            curves[L] = c
            cov = int(np.isfinite(c["eps_nd"]).sum())
            print(f"  L={L:3d}: {cov}/{grid.size} grid points covered")

    write_csv(prefix + "_lines.csv",
              ["eps_d"] + sum([[f"eps_nd_L{L}", f"sd_L{L}", f"mu_L{L}",
                                f"n_L{L}"] for L in sorted(curves)], []),
              [grid] + sum([[curves[L]["eps_nd"], curves[L]["eps_nd_sd"],
                             curves[L]["mu"], curves[L]["n"]]
                            for L in sorted(curves)], []))

    if len(curves) < 2:
        print("only one lattice size present -- no extrapolation")
        return 0

    ext = extrapolate(curves, nu=args.nu, nu_free=args.nu_free)
    write_csv(prefix + "_extrap.csv",
              ["eps_d", "eps_nd_inf", "slope_a", "rms_resid", "nu", "n_sizes"],
              [ext["eps_d"], ext["eps_nd_inf"], ext["slope"], ext["resid"],
               ext["nu"], ext["n_sizes"]])

    k = np.isfinite(ext["eps_nd_inf"])
    if k.any():
        print("\n" + "=" * 70)
        print(f"FINITE-SIZE EXTRAPOLATION  system={args.system}  "
              f"sizes={sorted(curves)}  nu={args.nu}")
        print(f"  eps_d covered      : {ext['eps_d'][k].min():.3f} .. "
              f"{ext['eps_d'][k].max():.3f}  ({int(k.sum())} points)")
        print(f"  RMS fit residual   : median "
              f"{np.nanmedian(ext['resid'][k]):.5f}, max "
              f"{np.nanmax(ext['resid'][k]):.5f}")
        for L in sorted(curves):
            d = curves[L]["eps_nd"] - ext["eps_nd_inf"]
            kk = np.isfinite(d)
            if kk.any():
                print(f"  Delta_FS(L={L:3d})   : mean "
                      f"{np.nanmean(d[kk]):+.5f}   max |.| "
                      f"{np.nanmax(np.abs(d[kk])):.5f}")
        print("=" * 70)

    if not args.no_figures:
        try:
            make_figures(curves, ext, prefix, args.system, rows)
        except Exception as exc:                               # noqa: BLE001
            print(f"[warn] figures failed: {exc}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
