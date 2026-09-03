#!/usr/bin/env python3
"""
Isomer (geometric-isomer) Monte Carlo for the SAFT-P manuscript.

Definitions
-----------
The lattice is half-filled with particles.  Family 1 is species 0..3 and
family 2 is species 4..7.  The instantaneous order parameter is

    m = (N2 - N1) / V,

with V=L^2.  Because N1+N2=V/2,

    Delta x = x2 - x1 = 2 m.

The three run modes have deliberately different meanings:

  wl
      Wang-Landau free-energy profile in N1.  The profile is reweighted to the
      field where the two basins have equal integrated weight.  This is the
      authoritative finite-L equilibrium binodal estimate.

  branch
      Seed the two perfect crystals and measure <m> at the user-specified
      fixed field --mu1.  This is a fixed-field diagnostic, NOT automatically
      a binodal measurement.

  branch-coex
      Seed the two perfect crystals at a coexistence field obtained from one or
      more matching WL JSON files supplied with --wl-source.  This is an
      independent validation of the WL binodal endpoints, while WL remains the
      definition of the equal-weight coexistence field.

Wang-Landau output
------------------
Converged output lng(N1) is ln Z(N1) at the field used in the WL run.  The
field enters only as a linear tilt in N1, so

    beta F(N1; mu) = -lng(N1) + mu_run*N1 - mu*N1 + const.

The analysis reports a two-phase validity flag.  Near/above criticality a
forced opposite-sign minimum can otherwise masquerade as a binodal; therefore
we explicitly check that both endpoints are local minima and that the smaller
escape barrier exceeds --min-barrier.
"""

import time, glob, json, os
import numpy as np
import numba as nb


def precompute_neighbors_flat(H, W):
    N = H * W
    nbr = np.empty((N, 4), dtype=np.int32)
    for i in range(H):
        im = (i - 1) % H; ip = (i + 1) % H
        for j in range(W):
            jm = (j - 1) % W; jp = (j + 1) % W
            k = i * W + j
            nbr[k,0]=im*W+j; nbr[k,1]=i*W+jp; nbr[k,2]=ip*W+j; nbr[k,3]=i*W+jm
    return nbr


def precompute_bond_table(J, patches):
    J = np.asarray(J, float); patches = np.asarray(patches, np.int16)
    ns = patches.shape[0]; B = np.empty((4, ns, ns))
    for si in range(ns):
        piN,piE,piS,piW = map(int, patches[si])
        for sj in range(ns):
            pjN,pjE,pjS,pjW = map(int, patches[sj])
            B[0,si,sj]=J[piN,pjS]; B[1,si,sj]=J[piE,pjW]
            B[2,si,sj]=J[piS,pjN]; B[3,si,sj]=J[piW,pjE]
    return B


@nb.njit(cache=True, fastmath=True)
def _push(buf, m, x):
    for i in range(m):
        if buf[i]==x: return m
    buf[m]=x; return m+1


@nb.njit(cache=True, fastmath=True)
def _eloc(k, lat, nbr, B):
    s=lat[k]
    return 0.5*(B[0,s,lat[nbr[k,0]]]+B[1,s,lat[nbr[k,1]]]+B[2,s,lat[nbr[k,2]]]+B[3,s,lat[nbr[k,3]]])


@nb.njit(cache=True, fastmath=True)
def _add(loc,lst,n,k):
    if loc[k]!=-1: return n
    loc[k]=n; lst[n]=k; return n+1


@nb.njit(cache=True, fastmath=True)
def _rm(loc,lst,n,k):
    i=loc[k]
    if i==-1: return n
    last=lst[n-1]; lst[i]=last; loc[last]=i; loc[k]=-1; return n-1


def species_patches():
    lut={'0':0}; nxt=1; out=[]
    for s in ['ABEF','BAEF','SSSS']:
        base=np.zeros(4,int)
        for i,ch in enumerate(s):
            if ch not in lut: lut[ch]=nxt; nxt+=1
            base[4-len(s)+i]=lut[ch]
        seen=set()
        for r in range(4):
            rot=tuple(np.roll(base,-r).tolist())
            if rot not in seen: seen.add(rot); out.append(np.array(rot))
    return np.array(out)


def K_matrix():
    # Kept exactly as in the submitted/production model.
    J=np.zeros((6,6)); J[1,3]=-3; J[2,4]=-3; J[3,3]=-1; J[4,4]=-1
    return J+J.T


def cand07():
    c=np.empty((8,7),np.int16)
    for s in range(8):
        t=0
        for x in range(8):
            if x!=s: c[s,t]=x; t+=1
    return c


@nb.njit(cache=True, fastmath=True)
def wl_kernel(lat0, nbr, B, mu, c07, lng, f0, fmin, flat_frac, min_steps, max_steps, seed):
    """Wang-Landau in N1 = number of family-1 particles (species 0..3)."""
    np.random.seed(seed)
    N = lat0.shape[0]
    lat = lat0.copy()
    NB = lng.shape[0]

    n1 = 0
    for k in range(N):
        if lat[k] < 4: n1 += 1

    gloc = np.full(N, -1, np.int32); glist = np.empty(N, np.int32); gn = 0
    for k in range(N):
        if lat[k] < 8:
            gloc[k] = gn; glist[gn] = k; gn += 1
    esite = np.empty(N)
    for k in range(N): esite[k] = _eloc(k, lat, nbr, B)

    buf10 = np.empty(10, np.int32); old10 = np.empty(10)
    buf5  = np.empty(5, np.int32); old5  = np.empty(5)
    H = np.zeros(NB, np.int64)

    f = f0
    while f > fmin:
        for b in range(NB): H[b] = 0
        steps = 0
        while True:
            steps += 1
            if np.random.rand() < 0.5:
                k1 = np.random.randint(N); k2 = np.random.randint(N-1)
                if k2 >= k1: k2 += 1
                s1 = lat[k1]; s2 = lat[k2]
                if s1 != s2:
                    m = 0; m = _push(buf10, m, k1); m = _push(buf10, m, k2)
                    for d in range(4): m = _push(buf10, m, nbr[k1, d])
                    for d in range(4): m = _push(buf10, m, nbr[k2, d])
                    for i in range(m): old10[i] = esite[buf10[i]]
                    lat[k1] = s2; lat[k2] = s1
                    dE = 0.0
                    for i in range(m):
                        kk = buf10[i]; v = _eloc(kk, lat, nbr, B); dE += v - old10[i]; esite[kk] = v
                    if dE <= 0.0 or np.random.rand() < np.exp(-dE):
                        if lat[k1] < 8: gn = _add(gloc, glist, gn, k1)
                        else:           gn = _rm(gloc, glist, gn, k1)
                        if lat[k2] < 8: gn = _add(gloc, glist, gn, k2)
                        else:           gn = _rm(gloc, glist, gn, k2)
                    else:
                        lat[k1] = s1; lat[k2] = s2
                        for i in range(m): esite[buf10[i]] = old10[i]
            else:
                if gn > 0:
                    k = glist[np.random.randint(gn)]; s = lat[k]
                    if s < 8:
                        new_s = c07[s, np.random.randint(7)]
                        n1_new = n1
                        if s < 4 and new_s >= 4: n1_new = n1 - 1
                        elif s >= 4 and new_s < 4: n1_new = n1 + 1
                        m = 0; m = _push(buf5, m, k)
                        for d in range(4): m = _push(buf5, m, nbr[k, d])
                        for i in range(m): old5[i] = esite[buf5[i]]
                        lat[k] = new_s
                        dE = 0.0
                        for i in range(m):
                            kk = buf5[i]; v = _eloc(kk, lat, nbr, B); dE += v - old5[i]; esite[kk] = v
                        dPhi = dE - (mu[new_s] - mu[s]) + (lng[n1_new] - lng[n1])
                        if dPhi <= 0.0 or np.random.rand() < np.exp(-dPhi):
                            n1 = n1_new
                        else:
                            lat[k] = s
                            for i in range(m): esite[buf5[i]] = old5[i]
            lng[n1] += f
            H[n1] += 1
            if steps >= min_steps:
                hmin = H[0]; hsum = 0.0
                for b in range(NB):
                    if H[b] < hmin: hmin = H[b]
                    hsum += H[b]
                if hmin > flat_frac * (hsum / NB):
                    break
                if steps >= max_steps:
                    break
        f = 0.5 * f
    return lng, H


def run_wl(T, mu1=0.042, seed=1, f0=1.0, fmin=1e-6, flat=0.75,
           min_steps=2_000_000, max_steps=60_000_000, L=20):
    P = species_patches(); K = K_matrix()
    nbr = precompute_neighbors_flat(L, L); c07 = cand07()
    B = precompute_bond_table(K / T, P)
    mu = np.zeros(9); mu[:4] = mu1
    N = L * L
    rng = np.random.default_rng(seed)
    lat = np.full(N, 8, np.int16)
    idx = rng.permutation(N)[:N // 2]
    lat[idx] = rng.integers(0, 8, N // 2)
    lng = np.zeros(N // 2 + 1)
    t0 = time.time()
    lng, H = wl_kernel(lat, nbr, B, mu, c07, lng, f0, fmin, flat,
                       min_steps, max_steps, seed * 977 + 13)
    return lng, H, time.time() - t0


# ---------------------------------------------------------------- analysis --
def _is_local_min(F, i):
    n = len(F)
    if i == 0:
        return F[0] <= F[1]
    if i == n - 1:
        return F[-1] <= F[-2]
    return F[i] <= F[i-1] and F[i] <= F[i+1]


def binodal_from_lng(lng, mu_run, L=20, min_barrier=0.5):
    """Equal-basin-weight coexistence field and finite-L binodal endpoints.

    The equal-weight field is found from the two basins separated by the free-
    energy maximum between opposite-sign minima.  The returned ``valid`` flag
    prevents a near/single-phase profile from being silently reported as a
    binodal.  ``barrier`` is the smaller of the two escape barriers.
    """
    from scipy.optimize import brentq

    lng = np.asarray(lng, float)
    NP = L * L // 2
    if len(lng) != NP + 1:
        raise ValueError(f"lng has {len(lng)} bins, expected {NP+1} for L={L}")

    N1 = np.arange(NP + 1)
    mg = 0.5 - N1 / float(NP)       # m=(N2-N1)/V
    F0 = -lng + mu_run * N1         # profile at mu1=0

    def split(mu):
        F = F0 - mu * N1
        F = F - F.min()
        # Stable exponentiation even if a trial tilt is large.
        p = np.exp(-np.minimum(F, 745.0))

        i1 = int(F.argmin())
        opp = np.where((mg * mg[i1]) < 0)[0]
        if opp.size == 0:
            raise ValueError("No opposite-sign order-parameter sector exists")
        i2 = int(opp[np.argmin(F[opp])])

        a, b = sorted([i1, i2])
        if b <= a:
            raise ValueError("Degenerate basin indices")
        top = int(a + np.argmax(F[a:b + 1]))

        # Split the barrier bin equally so it is not double-counted.
        w_left = p[:top].sum() + 0.5 * p[top]
        w_right = p[top+1:].sum() + 0.5 * p[top]
        return F, i1, i2, top, float(w_left), float(w_right)

    def log_weight_ratio(mu):
        _, _, _, _, wl, wr = split(mu)
        return np.log(wl / wr)

    # Expand the coexistence-field bracket if needed.
    lo, hi = -2.0, 2.0
    flo, fhi = log_weight_ratio(lo), log_weight_ratio(hi)
    for _ in range(12):
        if np.sign(flo) != np.sign(fhi):
            break
        lo *= 2.0; hi *= 2.0
        flo, fhi = log_weight_ratio(lo), log_weight_ratio(hi)
    if np.sign(flo) == np.sign(fhi):
        raise RuntimeError("Could not bracket the equal-basin-weight coexistence field")

    mu_c = brentq(log_weight_ratio, lo, hi, xtol=1e-10)
    F, i1, i2, top, w_left, w_right = split(mu_c)

    # Sort endpoints by Delta x = 2m, not by seed identity.
    vals = sorted([(2.0*mg[i1], i1), (2.0*mg[i2], i2)], key=lambda z: z[0])
    dx_minus, im = vals[0]
    dx_plus, ip = vals[1]

    b1 = float(F[top] - F[i1])
    b2 = float(F[top] - F[i2])
    barrier = min(b1, b2)
    local1 = bool(_is_local_min(F, i1))
    local2 = bool(_is_local_min(F, i2))
    opposite = bool(mg[i1] * mg[i2] < 0)
    valid = bool(opposite and local1 and local2 and barrier >= float(min_barrier))

    reasons = []
    if not opposite: reasons.append("endpoints_not_opposite_sign")
    if not local1 or not local2: reasons.append("endpoint_not_local_minimum")
    if barrier < float(min_barrier): reasons.append("barrier_below_threshold")

    return dict(
        mu_coex=float(mu_c),
        dx_minus=float(dx_minus), dx_plus=float(dx_plus),
        asymmetry=float(abs(dx_minus) - abs(dx_plus)),
        barrier=float(barrier), barrier_from_min1=b1, barrier_from_min2=b2,
        valid=valid, invalid_reason=";".join(reasons), min_barrier=float(min_barrier),
        basin_weight_left=w_left, basin_weight_right=w_right,
        F=F, dx_grid=2.0*mg,
    )


# ---------------------------------------------------------------- production kernel --
@nb.njit(cache=True, fastmath=True)
def kernel(lat0, nbr, B, mu, cand07, nsteps, nburn, interval, p_swap, seed):
    np.random.seed(seed)
    N=lat0.shape[0]; ns=mu.shape[0]; invA=1.0/N
    lat=lat0.copy()
    counts=np.zeros(ns,np.int64)
    for k in range(N): counts[lat[k]]+=1
    gloc=np.full(N,-1,np.int32); glist=np.empty(N,np.int32); gn=0
    for k in range(N):
        if lat[k]<8:
            gloc[k]=gn; glist[gn]=k; gn+=1
    esite=np.empty(N); energy=0.0
    for k in range(N):
        v=_eloc(k,lat,nbr,B); esite[k]=v; energy+=v
    buf10=np.empty(10,np.int32); old10=np.empty(10); buf5=np.empty(5,np.int32); old5=np.empty(5)
    tail=nsteps-nburn
    nsamp=tail//interval+1
    Es=np.empty(nsamp); Ms=np.empty(nsamp); si=0
    for t in range(nsteps+1):
        if np.random.rand()<p_swap:
            k1=np.random.randint(N); k2=np.random.randint(N-1)
            if k2>=k1: k2+=1
            s1=lat[k1]; s2=lat[k2]
            if s1!=s2:
                m=0; m=_push(buf10,m,k1); m=_push(buf10,m,k2)
                for d in range(4): m=_push(buf10,m,nbr[k1,d])
                for d in range(4): m=_push(buf10,m,nbr[k2,d])
                for i in range(m): old10[i]=esite[buf10[i]]
                lat[k1]=s2; lat[k2]=s1
                dE=0.0
                for i in range(m):
                    kk=buf10[i]; v=_eloc(kk,lat,nbr,B); dE+=v-old10[i]; esite[kk]=v
                if dE<=0.0 or np.random.rand()<np.exp(-dE):
                    energy+=dE
                    if lat[k1]<8: gn=_add(gloc,glist,gn,k1)
                    else: gn=_rm(gloc,glist,gn,k1)
                    if lat[k2]<8: gn=_add(gloc,glist,gn,k2)
                    else: gn=_rm(gloc,glist,gn,k2)
                else:
                    lat[k1]=s1; lat[k2]=s2
                    for i in range(m): esite[buf10[i]]=old10[i]
        else:
            if gn>0:
                k=glist[np.random.randint(gn)]; s=lat[k]
                if s<8:
                    new_s=cand07[s,np.random.randint(7)]
                    m=0; m=_push(buf5,m,k)
                    for d in range(4): m=_push(buf5,m,nbr[k,d])
                    for i in range(m): old5[i]=esite[buf5[i]]
                    lat[k]=new_s
                    dE=0.0
                    for i in range(m):
                        kk=buf5[i]; v=_eloc(kk,lat,nbr,B); dE+=v-old5[i]; esite[kk]=v
                    dPhi=dE-(mu[new_s]-mu[s])
                    if dPhi<=0.0 or np.random.rand()<np.exp(-dPhi):
                        energy+=dE; counts[s]-=1; counts[new_s]+=1
                    else:
                        lat[k]=s
                        for i in range(m): esite[buf5[i]]=old5[i]
        if t>=nburn and (t-nburn)%interval==0 and si<nsamp:
            Es[si]=energy
            n1=counts[0]+counts[1]+counts[2]+counts[3]
            n2=counts[4]+counts[5]+counts[6]+counts[7]
            Ms[si]=(n2-n1)*invA
            si+=1
    return Es[:si], Ms[:si], lat


# =====================================================================
# branch modes
# =====================================================================
GS = {"iso1": np.array([[0]]),
      "iso2": np.array([[4, 7], [5, 6]])}


def slab_lattice(which, L):
    """Half the box filled with the perfect crystal of `which`, half solvent."""
    tile = GS[which]
    lat = np.full((L, L), 8, np.int16)
    r, c = tile.shape
    reps = np.tile(tile, (max(1, (L // 2) // r + 1), max(1, L // c + 1)))
    lat[:L // 2, :] = reps[:L // 2, :L]
    return lat.ravel()


def run_branch(T, L, mu1, seed, steps, burn, interval):
    """Measure each seeded branch at a specified field.

    The returned ``m_sem`` is the naive within-trajectory SEM.  For reported
    uncertainties, prefer seed-to-seed SEM from mc_isomer_collect_coex.py.
    """
    P = species_patches(); K = K_matrix()
    nbr = precompute_neighbors_flat(L, L); c07 = cand07()
    B = precompute_bond_table(K / T, P)
    mu = np.zeros(9); mu[:4] = mu1
    out = {}
    for which in ("iso1", "iso2"):
        E, M, lat = kernel(slab_lattice(which, L), nbr, B, mu, c07,
                           int(steps), int(burn), int(interval), 0.5, seed * 7919 + 11)
        counts = np.bincount(lat, minlength=9)
        nz = M[np.sign(M) != 0]
        crossings = int((np.diff(np.sign(nz)) != 0).sum()) if len(nz) > 1 else 0
        mmean = float(M.mean())
        out[which] = dict(
            m_mean=mmean, dx_mean=2.0*mmean,
            m_sem=float(M.std(ddof=1) / np.sqrt(len(M))) if len(M) > 1 else 0.0,
            m_min=float(M.min()), m_max=float(M.max()),
            crossings=crossings,
            E_mean=float(E.mean() * T / (L * L)),
            final_species=counts.tolist(), n_samples=int(len(M)),
        )
    return out


def branch_quality(br):
    m1 = br["iso1"]["m_mean"]
    m2 = br["iso2"]["m_mean"]
    crossings = br["iso1"]["crossings"] + br["iso2"]["crossings"]
    opposite = bool(m1 * m2 < 0)
    clean = bool(opposite and crossings == 0)
    if clean:
        status = "clean_two_basin"
    elif not opposite:
        status = "same_sign_or_single_phase"
    else:
        status = "crossing_contaminated"
    return clean, status


def resolve_mu_coex(wl_source, T, L, require_valid=True):
    """Read matching WL JSON(s) and return mean coexistence field and seed SEM."""
    if wl_source is None:
        raise ValueError("branch-coex requires --wl-source FILE_OR_DIRECTORY")
    if os.path.isdir(wl_source):
        files = glob.glob(os.path.join(wl_source, "*.json"))
    else:
        files = [wl_source]

    vals = []
    used = []
    for fn in files:
        try:
            d = json.load(open(fn))
        except Exception:
            continue
        if d.get("mode") != "wl":
            continue
        if abs(float(d.get("T", np.nan)) - float(T)) > 1e-10 or int(d.get("L", -1)) != int(L):
            continue
        if require_valid and d.get("valid") is False:
            continue
        if "mu_coex" in d and np.isfinite(float(d["mu_coex"])):
            vals.append(float(d["mu_coex"]))
            used.append(fn)
    if not vals:
        raise RuntimeError(f"No matching WL coexistence result found for T={T}, L={L} in {wl_source}")
    vals = np.asarray(vals, float)
    sem = float(vals.std(ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else 0.0
    return float(vals.mean()), sem, used


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=["wl", "branch", "branch-coex"], required=True)
    p.add_argument("--T", type=float, required=True)
    p.add_argument("--L", type=int, default=20)
    p.add_argument("--mu1", type=float, default=0.0,
                   help="Field for wl or fixed-field branch mode. Ignored in branch-coex.")
    p.add_argument("--wl-source", default=None,
                   help="WL JSON file or directory used by branch-coex to obtain mu_coex.")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--steps", type=float, default=2e8, help="branch modes: total MC moves")
    p.add_argument("--burn-frac", type=float, default=0.4)
    p.add_argument("--interval", type=int, default=50_000)
    p.add_argument("--wl-fmin", type=float, default=1e-6)
    p.add_argument("--wl-flat", type=float, default=0.8)
    p.add_argument("--wl-min-steps", type=float, default=4e6)
    p.add_argument("--wl-max-steps", type=float, default=8e8)
    p.add_argument("--min-barrier", type=float, default=0.5,
                   help="Minimum smaller escape barrier (kT) required to flag a WL result as two-phase valid.")
    p.add_argument("--outdir", default=".")
    a = p.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    t0 = time.time()

    mu_used = float(a.mu1)

    if a.mode == "wl":
        lng, H, dt = run_wl(a.T, mu1=mu_used, seed=a.seed, fmin=a.wl_fmin, flat=a.wl_flat,
                            min_steps=int(a.wl_min_steps), max_steps=int(a.wl_max_steps), L=a.L)
        r = binodal_from_lng(lng, mu_used, L=a.L, min_barrier=a.min_barrier)
        res = dict(
            mode="wl", role="equilibrium_binodal", T=a.T, T_star=a.T / 2, L=a.L,
            mu1=mu_used, seed=a.seed, seconds=dt,
            mu_coex=r["mu_coex"], dx_minus=r["dx_minus"], dx_plus=r["dx_plus"],
            asymmetry=r["asymmetry"], barrier=r["barrier"],
            valid=r["valid"], invalid_reason=r["invalid_reason"], min_barrier=r["min_barrier"],
            basin_weight_left=r["basin_weight_left"], basin_weight_right=r["basin_weight_right"],
            lng=lng.tolist(),
        )
        flag = "VALID" if r["valid"] else f"INVALID:{r['invalid_reason']}"
        print(f"wl T={a.T} L={a.L} seed={a.seed}: Dx={r['dx_minus']:+.3f}/{r['dx_plus']:+.3f} "
              f"asym={r['asymmetry']:+.3f} mu_coex={r['mu_coex']:+.5f} "
              f"barrier={r['barrier']:.2f}kT {flag} ({dt:.0f}s)", flush=True)

    else:
        mu_source_files = []
        mu_source_sem = 0.0
        if a.mode == "branch-coex":
            mu_used, mu_source_sem, mu_source_files = resolve_mu_coex(a.wl_source, a.T, a.L)

        burn = int(a.steps * a.burn_frac)
        br = run_branch(a.T, a.L, mu_used, a.seed, a.steps, burn, a.interval)
        m1, m2 = br["iso1"]["m_mean"], br["iso2"]["m_mean"]
        clean, status = branch_quality(br)
        role = "coexistence_field_validation" if a.mode == "branch-coex" else "fixed_field_diagnostic"
        res = dict(
            mode=a.mode, role=role, T=a.T, T_star=a.T / 2, L=a.L, mu1=mu_used, seed=a.seed,
            steps=a.steps, seconds=time.time() - t0,
            m_asymmetry=abs(m1) - abs(m2), dx_asymmetry=2.0*(abs(m1) - abs(m2)),
            branch_clean=clean, branch_status=status,
            mu_source_sem=mu_source_sem, mu_source_files=mu_source_files,
            **br,
        )
        print(f"{a.mode} T={a.T} L={a.L} mu1={mu_used:+.5f} seed={a.seed}: "
              f"Dx(iso1)={2*m1:+.4f} Dx(iso2)={2*m2:+.4f} "
              f"Dx asym={2*(abs(m1)-abs(m2)):+.4f} "
              f"crossings={br['iso1']['crossings']}/{br['iso2']['crossings']} {status} "
              f"({time.time()-t0:.0f}s)", flush=True)

    fn = os.path.join(a.outdir,
                      f"{a.mode}_T{a.T:.2f}_L{a.L}_mu{mu_used:.5f}_s{a.seed}.json")
    with open(fn, "w") as fh:
        json.dump(res, fh)
    print("wrote", fn, flush=True)
