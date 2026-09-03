#!/usr/bin/env python3
"""Collect the revised Fig. 7 isomer MC campaign.

The collector deliberately separates:
  * fixed-field seeded branches -> diagnostic only;
  * branch-coex seeded runs      -> validation at WL coexistence field;
  * Wang-Landau                  -> equilibrium finite-L binodal.

Optional L->infinity extrapolation uses ONLY valid WL binodal endpoints at each
system size.  It never extrapolates fixed-field branch means as a binodal.
"""

import argparse, glob, json, os, csv
from collections import defaultdict
import numpy as np

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--workdir", required=True)
p.add_argument("--csv-prefix", default=None)
p.add_argument("--plot-prefix", default=None,
               help="If supplied, write equilibrium WL finite-size/error-bar plots.")
p.add_argument("--min-linf-sizes", type=int, default=3,
               help="Minimum number of distinct valid WL sizes required for an L->infinity fit.")
p.add_argument("--include-invalid-wl", action="store_true",
               help="Include WL JSONs explicitly flagged invalid (not recommended).")
a = p.parse_args()

branch_fixed = defaultdict(list)
branch_coex = defaultdict(list)
wl = defaultdict(list)

for fn in glob.glob(os.path.join(a.workdir, "*.json")):
    try:
        d = json.load(open(fn))
    except Exception:
        continue
    mode = d.get("mode")
    if mode == "branch":
        branch_fixed[(d["T"], d["L"], d["mu1"])].append(d)
    elif mode == "branch-coex":
        branch_coex[(d["T"], d["L"])].append(d)
    elif mode == "wl":
        if a.include_invalid_wl or d.get("valid", True):
            wl[(d["T"], d["L"])].append(d)


def ms(v):
    v = np.asarray(v, float)
    return v.mean(), (v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0.0)


def branch_stats(ds):
    m1, e1 = ms([d["iso1"]["m_mean"] for d in ds])
    m2, e2 = ms([d["iso2"]["m_mean"] for d in ds])
    dm = 2.0*m1; dp = 2.0*m2
    edm = 2.0*e1; edp = 2.0*e2
    dasym, edasym = ms([d.get("dx_asymmetry", 2.0*d.get("asymmetry", 0.0)) for d in ds])
    cr = sum(d["iso1"]["crossings"] + d["iso2"]["crossings"] for d in ds)
    opposite = (m1*m2 < 0)
    clean = opposite and cr == 0
    return dict(m1=m1,e1=e1,m2=m2,e2=e2,dm=dm,edm=edm,dp=dp,edp=edp,
                asym=dasym,easym=edasym,cross=cr,clean=clean)


def print_branch_table(title, data, has_fixed_mu):
    if not data:
        return
    print(title)
    if has_fixed_mu:
        print(f"{'T':>5} {'T*':>5} {'L':>4} {'mu1':>9} {'n':>3} | {'Dx(iso1)':>19} {'Dx(iso2)':>19} | {'Dx asym':>17} | {'cross':>6} {'quality':>12}")
        print("-"*126)
    else:
        print(f"{'T':>5} {'T*':>5} {'L':>4} {'mu_coex':>9} {'n':>3} | {'Dx(iso1)':>19} {'Dx(iso2)':>19} | {'Dx asym':>17} | {'cross':>6} {'quality':>12}")
        print("-"*126)
    for k in sorted(data):
        ds = data[k]
        st = branch_stats(ds)
        if has_fixed_mu:
            T,L,mu = k
        else:
            T,L = k
            mu,_ = ms([d["mu1"] for d in ds])
        q = "clean" if st["clean"] else "NOT_BINODAL"
        print(f"{T:5.2f} {T/2:5.2f} {L:4d} {mu:+9.5f} {len(ds):3d} | "
              f"{st['dm']:+9.4f} +/- {st['edm']:.4f} {st['dp']:+9.4f} +/- {st['edp']:.4f} | "
              f"{st['asym']:+8.4f} +/- {st['easym']:.4f} | {st['cross']:6d} {q:>12}")
    print()


print_branch_table(
    "FIXED-FIELD SEEDED BRANCHES  [diagnostic only; not an equilibrium binodal unless the field is independently known to be mu_coex]",
    branch_fixed, True)
print_branch_table(
    "COEX-TUNED SEEDED BRANCHES  [validation runs at the WL equal-weight coexistence field]",
    branch_coex, False)


# Aggregate valid WL results by (T,L).
wl_agg = {}
if wl:
    print("WANG-LANDAU  [authoritative finite-L equilibrium binodal: equal basin weight]")
    print(f"{'T':>5} {'T*':>5} {'L':>4} {'n':>3} | {'Dx(-)':>18} {'Dx(+)':>18} | {'asym':>18} | {'mu_coex':>18} {'barrier':>9}")
    print("-"*125)
    for k in sorted(wl):
        T,L = k; ds = wl[k]
        dm, em = ms([d["dx_minus"] for d in ds])
        dp, ep = ms([d["dx_plus"] for d in ds])
        asym, ea = ms([d.get("asymmetry", abs(d["dx_minus"])-abs(d["dx_plus"])) for d in ds])
        mu, emu = ms([d["mu_coex"] for d in ds])
        ba, eba = ms([d["barrier"] for d in ds])
        wl_agg[k] = dict(T=T,L=L,n=len(ds),dm=dm,em=em,dp=dp,ep=ep,
                         asym=asym,ea=ea,mu=mu,emu=emu,barrier=ba,ebarrier=eba)
        print(f"{T:5.2f} {T/2:5.2f} {L:4d} {len(ds):3d} | "
              f"{dm:+8.3f} +/- {em:.3f} {dp:+8.3f} +/- {ep:.3f} | "
              f"{asym:+8.3f} +/- {ea:.3f} | {mu:+8.5f} +/- {emu:.5f} {ba:9.2f}")
    print()


def wls_intercept_invL(rows, value_key, se_key):
    L = np.asarray([r["L"] for r in rows], float)
    x = 1.0/L
    y = np.asarray([r[value_key] for r in rows], float)
    se = np.asarray([r[se_key] for r in rows], float)

    pos = se[np.isfinite(se) & (se > 0)]
    floor = max(1e-4, (np.median(pos)*0.25 if len(pos) else 1e-3))
    sigma = np.where(np.isfinite(se) & (se > 0), np.maximum(se, floor), floor)
    w = 1.0/sigma**2
    A = np.column_stack([np.ones_like(x), x])
    Aw = A*np.sqrt(w)[:,None]
    yw = y*np.sqrt(w)
    beta, *_ = np.linalg.lstsq(Aw, yw, rcond=None)
    cov = np.linalg.inv(Aw.T@Aw)
    intercept, slope = beta
    se_intercept, se_slope = np.sqrt(np.diag(cov))
    fit = A@beta
    rms = float(np.sqrt(np.mean((y-fit)**2)))
    return dict(intercept=float(intercept), se=float(se_intercept), slope=float(slope),
                slope_se=float(se_slope), rms=rms)


# L->infinity fits use only WL endpoints.
linf = []
if wl_agg:
    byT = defaultdict(list)
    for (T,L), r in wl_agg.items():
        byT[T].append(r)

    print("WL FINITE-SIZE EXTRAPOLATION  [fit each equilibrium endpoint vs 1/L]")
    print(f"{'T':>5} {'T*':>5} {'nL':>3} | {'Dx(-,inf)':>20} {'Dx(+,inf)':>20} | {'asym_inf':>20} | {'status':>11}")
    print("-"*112)
    for T in sorted(byT):
        rows = sorted(byT[T], key=lambda r:r["L"])
        nL = len(rows)
        if nL < a.min_linf_sizes:
            print(f"{T:5.2f} {T/2:5.2f} {nL:3d} | {'--':>20} {'--':>20} | {'--':>20} | need >={a.min_linf_sizes} L")
            continue
        fm = wls_intercept_invL(rows, "dm", "em")
        fp = wls_intercept_invL(rows, "dp", "ep")
        asym = abs(fm["intercept"]) - abs(fp["intercept"])
        easym = np.sqrt(fm["se"]**2 + fp["se"]**2)
        status = "exploratory" if nL < 4 else "usable"
        rec = dict(T=T,T_star=T/2,n_sizes=nL,dx_minus_inf=fm["intercept"],se_minus_inf=fm["se"],
                   dx_plus_inf=fp["intercept"],se_plus_inf=fp["se"],asym_inf=asym,se_asym_inf=easym,
                   rms_minus=fm["rms"],rms_plus=fp["rms"],status=status)
        linf.append(rec)
        print(f"{T:5.2f} {T/2:5.2f} {nL:3d} | {fm['intercept']:+9.4f} +/- {fm['se']:.4f} "
              f"{fp['intercept']:+9.4f} +/- {fp['se']:.4f} | {asym:+9.4f} +/- {easym:.4f} | {status:>11}")
    print()


if a.csv_prefix:
    # Fixed-field branch CSV
    with open(a.csv_prefix + "_branch_fixed.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["T","T_star","L","mu1","seed","m_iso1","m_iso2","Dx_iso1","Dx_iso2","Dx_asymmetry","crossings","role"])
        for k in sorted(branch_fixed):
            for d in branch_fixed[k]:
                m1=d["iso1"]["m_mean"]; m2=d["iso2"]["m_mean"]
                cr=d["iso1"]["crossings"]+d["iso2"]["crossings"]
                w.writerow([d["T"],d["T_star"],d["L"],d["mu1"],d["seed"],m1,m2,2*m1,2*m2,2*(abs(m1)-abs(m2)),cr,"diagnostic"])

    with open(a.csv_prefix + "_branch_coex.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["T","T_star","L","mu_coex","seed","m_iso1","m_iso2","Dx_iso1","Dx_iso2","Dx_asymmetry","crossings","role"])
        for k in sorted(branch_coex):
            for d in branch_coex[k]:
                m1=d["iso1"]["m_mean"]; m2=d["iso2"]["m_mean"]
                cr=d["iso1"]["crossings"]+d["iso2"]["crossings"]
                w.writerow([d["T"],d["T_star"],d["L"],d["mu1"],d["seed"],m1,m2,2*m1,2*m2,2*(abs(m1)-abs(m2)),cr,"coex_validation"])

    with open(a.csv_prefix + "_wl.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["T","T_star","L","seed","dx_minus","dx_plus","asymmetry","mu_coex","barrier","valid"])
        for k in sorted(wl):
            for d in wl[k]:
                w.writerow([d["T"],d["T_star"],d["L"],d["seed"],d["dx_minus"],d["dx_plus"],
                            d.get("asymmetry",abs(d["dx_minus"])-abs(d["dx_plus"])),d["mu_coex"],d["barrier"],d.get("valid",True)])

    with open(a.csv_prefix + "_wl_linf.csv", "w", newline="") as fh:
        if linf:
            w = csv.DictWriter(fh, fieldnames=list(linf[0].keys()))
            w.writeheader(); w.writerows(linf)
        else:
            fh.write("T,T_star,n_sizes,dx_minus_inf,se_minus_inf,dx_plus_inf,se_plus_inf,asym_inf,se_asym_inf,rms_minus,rms_plus,status\n")
    print(f"wrote {a.csv_prefix}_branch_fixed.csv, {a.csv_prefix}_branch_coex.csv, {a.csv_prefix}_wl.csv, {a.csv_prefix}_wl_linf.csv")


if a.plot_prefix and wl_agg:
    import matplotlib.pyplot as plt

    # Plot A: equilibrium WL binodals by finite L, plus L->inf where available.
    fig, ax = plt.subplots(figsize=(7.4,5.6))
    sizes = sorted({r["L"] for r in wl_agg.values()})
    for L in sizes:
        rows = sorted([r for r in wl_agg.values() if r["L"]==L], key=lambda r:r["T"])
        Ts = np.array([r["T"]/2 for r in rows])
        dm = np.array([r["dm"] for r in rows]); em=np.array([r["em"] for r in rows])
        dp = np.array([r["dp"] for r in rows]); ep=np.array([r["ep"] for r in rows])
        ax.errorbar(dm, Ts, xerr=em, fmt='o--', capsize=3, label=f"WL L={L}")
        ax.errorbar(dp, Ts, xerr=ep, fmt='o--', capsize=3)
    if linf:
        Ts=np.array([r["T_star"] for r in linf]); dm=np.array([r["dx_minus_inf"] for r in linf]); em=np.array([r["se_minus_inf"] for r in linf])
        dp=np.array([r["dx_plus_inf"] for r in linf]); ep=np.array([r["se_plus_inf"] for r in linf])
        ax.errorbar(dm, Ts, xerr=em, fmt='s-', capsize=3, label=r"WL $L\to\infty$")
        ax.errorbar(dp, Ts, xerr=ep, fmt='s-', capsize=3)
    ax.set_xlabel(r"Isomer excess, $\Delta x=x_2-x_1$")
    ax.set_ylabel(r"Normalized temperature, $T^*$")
    ax.tick_params(direction='in',top=True,right=True)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(a.plot_prefix + "_wl_binodal.png", dpi=300)
    fig.savefig(a.plot_prefix + "_wl_binodal.pdf")
    plt.close(fig)

    # Plot B: asymmetry vs 1/L, one curve per T.
    fig, ax = plt.subplots(figsize=(7.4,5.6))
    byT=defaultdict(list)
    for r in wl_agg.values(): byT[r["T"]].append(r)
    for T in sorted(byT):
        rows=sorted(byT[T],key=lambda r:r["L"])
        x=np.array([1/r["L"] for r in rows]); y=np.array([r["asym"] for r in rows]); e=np.array([r["ea"] for r in rows])
        ax.errorbar(x,y,yerr=e,fmt='o',capsize=3,label=fr"$T^*={T/2:.2f}$")
        if len(rows)>=a.min_linf_sizes:
            fit=wls_intercept_invL(rows,"asym","ea")
            xx=np.linspace(0,x.max()*1.03,100); ax.plot(xx,fit["intercept"]+fit["slope"]*xx)
    ax.axhline(0,lw=1)
    ax.set_xlabel(r"$1/L$")
    ax.set_ylabel(r"Equilibrium binodal asymmetry, $|\Delta x_-|-|\Delta x_+|$")
    ax.tick_params(direction='in',top=True,right=True)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(a.plot_prefix + "_wl_asymmetry_invL.png", dpi=300)
    fig.savefig(a.plot_prefix + "_wl_asymmetry_invL.pdf")
    plt.close(fig)
    print(f"wrote {a.plot_prefix}_wl_binodal.[png,pdf] and {a.plot_prefix}_wl_asymmetry_invL.[png,pdf]")
