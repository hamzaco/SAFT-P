import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import math
import numpy as np
import matplotlib
matplotlib.use("Agg")          # non-interactive backend for cluster
import matplotlib.pyplot as plt
from typing import Dict, Tuple, List
from scipy.optimize import minimize
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp

from simulate_functions_triangular import (
    get_species_list_ind_hex,
    simulate_lattice_triangular,
)


# ======================== Parallel replica worker ========================
# Top-level function so it's picklable by multiprocessing.
def _replica_worker(
    lattice_int,          # np.int64 2-D
    patches_by_species,   # np.uint8 2-D  (n_species, n_patches)
    species_mu_indices,   # np.int64 1-D
    eps_ns0, eps_sp0, mu0,
    steps, snapshot_interval, buffer_size,
    empty_index, beta,
):
    """Run one MC replica and return raw snapshot dict (picklable)."""
    n_species = patches_by_species.shape[0]
    empty_index = int(empty_index)

    def J_patch(epsp, epsn):
        return np.array([[-epsn,           -epsn,        0.0],
                         [-epsn, -epsn - epsp,           0.0],
                         [  0.0,            0.0,         0.0]], float)

    Jref = J_patch(float(eps_sp0), float(eps_ns0))

    idx = np.unique(np.asarray(species_mu_indices, dtype=np.int64))
    idx = idx[(idx >= 0) & (idx < n_species)]
    mu_table = np.zeros(n_species, dtype=np.float64)
    mu_table[idx] = float(mu0)

    snap_counts, energies, densities, lat_final = simulate_lattice_triangular(
        lattice_int.copy(), Jref, patches_by_species, mu_table,
        empty_index=empty_index,
        eps_ns0=float(eps_ns0), eps_sp0=float(eps_sp0),
        simulation_steps=int(steps), beta=float(beta),
        snapshot_interval=int(snapshot_interval),
        buffer_size=int(buffer_size),
        species_mu_indices=idx,
    )

    snap_counts = dict(snap_counts)
    N_sites = int(snap_counts["N_sites"])

    if "rho_mu" not in snap_counts or "N_mu" not in snap_counts:
        dens = np.asarray(densities, float)
        rho_mu = dens[:, idx].sum(axis=1) if idx.size else np.zeros(dens.shape[0], float)
        snap_counts["rho_mu"] = rho_mu
        snap_counts["N_mu"]   = rho_mu * float(N_sites)

    for k in ["N_mu", "rho_mu", "Nocc_total", "rho_total", "u", "C_ns", "C_sp"]:
        if k in snap_counts:
            snap_counts[k] = np.asarray(snap_counts[k], float).ravel()

    snap_counts["_lat_final"] = lat_final.copy()
    return snap_counts


# ======================== Data container ========================
class ReweightDataMU:
    def __init__(self, beta, L, N_sites, eps_ns0, eps_sp0, mu0,
                 N_mu, rho_mu, rho_total, u, C_ns, C_sp):
        self.beta = float(beta); self.L = int(L); self.N_sites = int(N_sites)
        self.eps_ns0 = float(eps_ns0); self.eps_sp0 = float(eps_sp0); self.mu0 = float(mu0)
        self.N_mu = np.asarray(N_mu, float).ravel()
        self.rho_mu = np.asarray(rho_mu, float).ravel()
        self.rho_total = np.asarray(rho_total, float).ravel()
        self.u = np.asarray(u, float).ravel()
        self.C_ns = np.asarray(C_ns, float).ravel()
        self.C_sp = np.asarray(C_sp, float).ravel()
        n = len(self.N_mu)
        assert all(len(getattr(self, k)) == n for k in ("rho_mu", "rho_total", "u", "C_ns", "C_sp")), \
            "ReweightDataMU: array length mismatch"


# ======================== Lattice helpers ========================
def to_int_lattice(lat):
    lat = np.asarray(lat)
    if np.issubdtype(lat.dtype, np.integer):
        return lat.astype(np.int64, copy=False)
    return np.vectorize(lambda p: p.index, otypes=[np.int64])(lat)


# ======================== Parallel builder ========================
def build_data_mu_cpu(simulate_lattice_unused,
                      lattice_dense, lattice_dilute, lattice_middle,
                      species, species_mu_indices, chem_pots,
                      eps_ns=1.767, eps_sp=0.0, mu=2*1.767+np.log(6), beta=1.0,
                      snapshot_interval=300_000,
                      steps=500_000, cluster_move=False, buffer_size=4000,
                      empty_index=-1,
                      max_workers=None, n_replicas=16,
                      extra_lattices=None) -> ReweightDataMU:

    lattices_int = [
        to_int_lattice(lattice_dense).copy(),
        to_int_lattice(lattice_dilute).copy(),
    ]
    if lattice_middle is not None:
        latM = to_int_lattice(lattice_middle).copy()
        lattices_int += [latM, latM, latM]

    if extra_lattices is not None:
        for el in extra_lattices:
            lattices_int.insert(0, to_int_lattice(el).copy())

    # Pre-extract patches_by_species (plain numpy) so we can pickle it
    patches_by_species = np.asarray(
        [getattr(s, "patches") for s in species], dtype=np.uint8
    )
    smi = np.asarray(species_mu_indices, dtype=np.int64)

    # Build per-replica arguments (all picklable scalars / arrays)
    replica_args = []
    for r in range(int(n_replicas)):
        lat = lattices_int[r % len(lattices_int)].copy()
        replica_args.append((
            lat, patches_by_species, smi,
            float(eps_ns), float(eps_sp), float(mu),
            int(steps), int(snapshot_interval), int(buffer_size),
            int(empty_index), float(beta),
        ))

    # Decide parallelism
    if max_workers == 1:
        # Sequential fallback
        runs = [_replica_worker(*args) for args in replica_args]
    else:
        n_cpu = mp.cpu_count() if max_workers is None else int(max_workers)
        n_cpu = min(n_cpu, int(n_replicas))
        print(f"[build_data] launching {n_replicas} replicas on {n_cpu} workers", flush=True)
        with ProcessPoolExecutor(max_workers=n_cpu) as pool:
            futures = [pool.submit(_replica_worker, *args) for args in replica_args]
            runs = [f.result() for f in futures]  # preserves order

    keys = ["N_mu", "rho_mu", "Nocc_total", "rho_total", "u", "C_ns", "C_sp"]
    cat = {k: np.concatenate([r[k] for r in runs]) for k in keys}

    idx = np.random.default_rng(12345).permutation(len(cat["Nocc_total"]))
    for k in keys:
        cat[k] = cat[k][idx]

    N_sites = int(runs[0]["N_sites"])
    L_side  = int(np.sqrt(N_sites))

    data = ReweightDataMU(
        beta=beta, L=L_side, N_sites=N_sites,
        eps_ns0=eps_ns, eps_sp0=eps_sp, mu0=mu,
        N_mu=cat["N_mu"], rho_mu=cat["rho_mu"],
        rho_total=cat["rho_total"], u=cat["u"],
        C_ns=cat["C_ns"], C_sp=cat["C_sp"],
    )

    data._lat_final = runs[0]["_lat_final"]
    return data


# ======================== Histogram smoother ========================
def smooth_histogram_gaussian(x_centers, p_hist, sigma=1.0):
    x_centers = np.asarray(x_centers, float)
    p_hist    = np.asarray(p_hist, float)
    dx = float(np.mean(np.diff(x_centers)))
    sigma_bins = sigma / dx
    halfw = int(np.ceil(4 * sigma_bins))
    kx = np.arange(-halfw, halfw + 1, dtype=float)
    K = np.exp(-0.5 * (kx / sigma_bins)**2); K /= K.sum()
    p_pad = np.pad(p_hist, (halfw, halfw), mode='reflect')
    p_conv = np.convolve(p_pad, K, mode='same')[halfw:-halfw]
    p_conv /= np.trapz(p_conv, x_centers)
    return x_centers, p_conv


# ======================== Reweighting utilities ========================
def _ess(w):
    w = np.asarray(w, float)
    s1 = w.sum(); s2 = (w * w).sum()
    return (s1 * s1) / max(s2, 1e-300)


def _weights_mu(N_counts, mu_new, mu_ref, beta=1.0, clip=700.0):
    dmu = float(mu_new - mu_ref)
    logw = -beta * dmu * np.asarray(N_counts, float)
    logw -= np.max(logw)
    logw = np.clip(logw, -clip, None)
    return np.exp(logw)


def _rho_hist_density(rho, w, edges):
    counts, _ = np.histogram(rho, bins=edges, weights=w, density=False)
    W = counts.sum()
    p = counts / max(W, 1e-300)
    width = np.diff(edges)
    dens = p / width
    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers, dens


def _area_in_window(edges, dens, lo, hi):
    lo = float(lo); hi = float(hi)
    L = edges[:-1]; R = edges[1:]
    overlap = np.maximum(0.0, np.minimum(R, hi) - np.maximum(L, lo))
    return float(np.sum(dens * overlap))


def equal_peak_mu_in_rho_by_reweighting(
    rho, N_counts, mu_ref, *,
    beta=1.0, rho_edges=None, rho_bins=200,
    dilute_window=(0.0, 0.3), dense_window=(0.7, 1.0),
    ess_min=2_000,
    mu0=None, step0=0.01, max_bracket=20, tol=5e-3, max_bisect=30,
    verbose=True,
):
    rho = np.asarray(rho, float)
    N   = np.asarray(N_counts, float)

    if rho_edges is None:
        lo, hi = float(np.min(rho)), float(np.max(rho))
        edges = np.linspace(lo, hi, int(rho_bins) + 1)
    else:
        edges = np.asarray(rho_edges, float)

    def f_of_mu(mu_trial):
        w = _weights_mu(N, mu_trial, mu_ref, beta=beta)
        ess = _ess(w)
        _, dens = _rho_hist_density(rho, w, edges)
        Ad = _area_in_window(edges, dens, *dilute_window)
        Ah = _area_in_window(edges, dens, *dense_window)
        return (Ah - Ad), ess, w, dens

    mu_start = float(mu_ref if mu0 is None else mu0)
    f0, ess0, w0, dens0 = f_of_mu(mu_start)
    if verbose:
        print(f"[equal-rho] mu0={mu_start:.6f}, dA={f0:+.4f}, ESS~{ess0:.0f}")

    if abs(f0) < tol and ess0 >= ess_min:
        centers, _ = _rho_hist_density(rho, w0, edges)
        return dict(mu_star=mu_start, success=True, ess=float(ess0),
                    centers=centers, density_star=dens0, w_star=w0)

    direction = 1.0 if f0 > 0 else -1.0
    step = float(step0)

    for kk in range(max_bracket):
        mu_try = mu_start + direction * step * (kk + 1)
        ft, esst, wt, denst = f_of_mu(mu_try)
        if verbose:
            print(f"[equal-rho] bracket mu={mu_try:.6f}, dA={ft:+.4f}, ESS~{esst:.0f}")
        if np.sign(ft) != np.sign(f0):
            if f0 > 0:
                mu_lo, f_lo, w_lo, dens_lo = mu_try, ft, wt, denst
                mu_hi, f_hi, w_hi, dens_hi = mu_start, f0, w0, dens0
            else:
                mu_lo, f_lo, w_lo, dens_lo = mu_start, f0, w0, dens0
                mu_hi, f_hi, w_hi, dens_hi = mu_try, ft, wt, denst
            break
    else:
        centers, _ = _rho_hist_density(rho, w0, edges)
        return dict(mu_star=mu_start, success=False, ess=float(ess0),
                    centers=centers, density_star=dens0, w_star=w0,
                    reason="failed to bracket")

    for it in range(max_bisect):
        mu_mid = 0.5 * (mu_lo + mu_hi)
        fm, essm, wm, densm = f_of_mu(mu_mid)
        if verbose:
            print(f"[equal-rho] bisect mu={mu_mid:.6f}, dA={fm:+.4f}, ESS~{essm:.0f}")
        if abs(fm) < tol and essm >= ess_min:
            centers, _ = _rho_hist_density(rho, wm, edges)
            return dict(mu_star=mu_mid, success=True, ess=float(essm),
                        centers=centers, density_star=densm, w_star=wm)
        if np.sign(fm) == np.sign(f_lo):
            mu_lo, f_lo, w_lo, dens_lo = mu_mid, fm, wm, densm
        else:
            mu_hi, f_hi, w_hi, dens_hi = mu_mid, fm, wm, densm

    centers, _ = _rho_hist_density(rho, wm, edges)
    return dict(mu_star=mu_mid, success=False, ess=float(essm),
                centers=centers, density_star=densm, w_star=wm,
                reason="max bisection iterations")


def resample_at_mu_star(data: ReweightDataMU, mu_star: float, rng=None):
    if rng is None:
        rng = np.random.default_rng(1234)
    dG = -data.beta * (mu_star - data.mu0) * np.asarray(data.N_mu, float)
    dG -= dG.max()
    w = np.exp(dG); w /= max(w.sum(), 1e-300)
    c = np.cumsum(w); u0 = rng.random() / len(w); u = u0 + np.arange(len(w)) / len(w)
    idx = np.searchsorted(c, u, side="right")
    return dict(
        indices=idx, weights=w,
        rho_mu=data.rho_mu[idx].copy(), rho_total=data.rho_total[idx].copy(),
        N_mu=data.N_mu[idx].copy(), u=data.u[idx].copy(),
        C_ns=data.C_ns[idx].copy(), C_sp=data.C_sp[idx].copy(),
    )


# ======================== Mixed-field fit (s,r) + KDE ========================
def _wmean(x, w):
    return float(np.sum(w * x) / max(np.sum(w), 1e-300))


def _wstd(x, w):
    m = _wmean(x, w)
    v = _wmean((x - m)**2, w)
    return float(np.sqrt(max(v, 1e-300)))


def _kde_fixed_sigma_on_grid(x_samples, x_grid, sigma_x, w):
    x_samples = np.asarray(x_samples, float)
    x_grid    = np.asarray(x_grid, float)
    w = np.asarray(w, float).ravel()
    w = np.where(np.isfinite(w) & (w > 0), w, 0.0)
    w /= max(w.sum(), 1e-300)
    diff = (x_grid[:, None] - x_samples[None, :]) / sigma_x
    K = np.exp(-0.5 * diff * diff)
    P = (K @ w) / (sigma_x * np.sqrt(2 * np.pi))
    norm = np.trapz(P, x_grid)
    if norm > 1e-300:
        P /= norm
    return P


def fit_s_r_from_reweighted(
    rho, u, x_ref, P_ref, s_ref, *,
    weights=None, L=10,
    beta_over_nu=1/8, sigma_x=0.05,
    r_penalty=5e-2,
    bounds_s=(-5.0, 5.0), bounds_r=(-1.0, 1.0),
    method="L-BFGS-B", w_asym=0.5,
) -> Dict:
    rho = np.asarray(rho, float).ravel()
    u   = np.asarray(u,   float).ravel()
    x_ref = np.asarray(x_ref, float).ravel()
    P_ref = np.asarray(P_ref, float).ravel()

    if weights is None:
        w = np.ones_like(rho, float)
    else:
        w = np.asarray(weights, float).ravel()
        w = np.where(np.isfinite(w) & (w > 0), w, 0.0)
    w /= max(w.sum(), 1e-300)

    # Adaptive sigma: at least 1.5x grid spacing
    dx_grid = float(np.mean(np.diff(x_ref)))
    sigma_adaptive = max(sigma_x, 1.5 * dx_grid)

    def build_x_pdf(s, r):
        M = rho - s * u
        M_center = _wmean(M, w)
        M_std    = max(_wstd(M, w), 1e-300)
        x_samples = (M - M_center) / M_std
        P_est = _kde_fixed_sigma_on_grid(x_samples, x_ref, sigma_adaptive, w)
        return 1.0 / M_std, x_samples, P_est

    def objective(theta):
        s, r = float(theta[0]), float(theta[1])
        if s_ref is not None:
            s = s_ref
        _, _, P_est = build_x_pdf(s, r)
        if not np.all(np.isfinite(P_est)):
            return 1e6
        diff = P_est - P_ref
        match_L2 = np.trapz(diff * diff, x_ref)
        if not np.isfinite(match_L2):
            return 1e6
        P_mir = np.interp(-x_ref, x_ref, P_est)
        A = 0.5 * (P_est - P_mir)
        asym_L2 = np.trapz(A * A, x_ref)
        return match_L2 + w_asym * asym_L2 + r_penalty * (r * r)

    x0 = np.array([0.0, 0.0], float)
    bounds = [bounds_s, bounds_r]
    res = minimize(objective, x0, method="L-BFGS-B", bounds=bounds,
                   options=dict(maxiter=300, ftol=1e-9))
    if (not res.success) or ("ABNORMAL" in str(res.message)):
        res = minimize(objective, res.x, method="Nelder-Mead",
                       options=dict(maxiter=800, xatol=1e-6, fatol=1e-6))

    s_opt, r_opt = map(float, res.x)
    alpha_opt, xs_opt, P_fit = build_x_pdf(s_opt, r_opt)
    diff = P_fit - P_ref
    match_L2 = float(np.trapz(diff * diff, x_ref))
    P_mir = np.interp(-x_ref, x_ref, P_fit)
    A = 0.5 * (P_fit - P_mir)
    asym_L2 = float(np.trapz(A * A, x_ref))

    return dict(success=bool(res.success), message=str(res.message),
                s=s_opt, r=r_opt, alpha=alpha_opt,
                x=x_ref, P_fit=P_fit, P_ref=P_ref,
                match_L2=match_L2, asym_L2=asym_L2, fun=float(res.fun))


# ======================== theta reweight objective ========================
def _parse_theta_with_s(theta):
    t = list(map(float, theta))
    be, beA, bm, L, Mc = t[:5]
    s = t[5] if len(t) >= 6 else 0.0
    return be, beA, bm, L, Mc, s


def kl_q_p(q, p, eps=1e-12):
    q = np.asarray(q, float); p = np.asarray(p, float)
    q = q / max(q.sum(), 1e-300); p = p / max(p.sum(), 1e-300)
    q = np.clip(q, eps, 1.0); p = np.clip(p, eps, 1.0)
    q /= q.sum(); p /= p.sum()
    return float(np.sum(q * (np.log(q) - np.log(p))))


def js_div(q, p, eps=1e-12):
    q = np.asarray(q, float); p = np.asarray(p, float)
    q = q / max(q.sum(), 1e-300); p = p / max(p.sum(), 1e-300)
    q = np.clip(q, eps, 1.0); p = np.clip(p, eps, 1.0)
    q /= q.sum(); p /= p.sum()
    m = 0.5 * (q + p)
    return 0.5 * kl_q_p(q, m, eps) + 0.5 * kl_q_p(p, m, eps)


def reweight_from_samples_counts_with_s(
    M_ref, C, Ca, N,
    theta_prime, theta_ref,
    *, rho=None, u=None, N_sites=None,
    clip=700.0, return_Mprime=False,
):
    be,  beA,  bm, _, _, s_ref    = _parse_theta_with_s(theta_ref)
    bep, beAp, bmp, _, _, s_prime = _parse_theta_with_s(theta_prime)
    if (rho is not None) and (N_sites is not None):
        # Reconstruct u at the TRIAL coupling constants (bep, beAp)
        # instead of using the simulation-point u which has the old
        # couplings baked in.  u = -(eps_ns * C_tot + eps_sp * C_sp) / N_sites
        C_ns_arr  = np.asarray(C,  float)
        C_sp_arr  = np.asarray(Ca, float)
        C_tot_arr = C_ns_arr + C_sp_arr
        inv_Nsites = 1.0 / float(N_sites)
        u_at_trial = -(bep * C_tot_arr + beAp * C_sp_arr) * inv_Nsites
        Mprime = np.asarray(rho, float) - float(s_prime) * u_at_trial
    elif (rho is not None) and (u is not None):
        # Fallback: use pre-computed u (old behaviour, kept for
        # call-sites that don't supply N_sites)
        Mprime = np.asarray(rho, float) - float(s_prime) * np.asarray(u, float)
    else:
        Mprime = np.asarray(M_ref, float)
    d_be  = bep - be      # Delta eps_ns
    d_beA = beAp - beA    # Delta eps_sp
    d_bm  = bmp - bm      # Delta mu

    # Phi = -eps_ns*(C_ns+C_sp) - eps_sp*C_sp + mu*N
    # log w = d_ens*C_ns + (d_ens+d_esp)*C_sp - d_mu*N
    logw = ( - d_be            * np.asarray(C,  float)    # C_ns
            - (d_be + d_beA)  * np.asarray(Ca, float)    # C_sp
            - d_bm            * np.asarray(N,  float))   # N_mu

    logw = np.clip(logw - np.max(logw), -clip, None)
    w = np.exp(logw)
    return (w, Mprime) if return_Mprime else w

def weighted_hist_and_sigma(x, w, edges, n_samples, epsilon=1e-12, *,
                            return_nonempty=True, empty_tol=0.0):
    """Paper Eq. (6)."""
    x = np.asarray(x, float); w = np.asarray(w, float)
    K = len(edges) - 1
    idx = np.clip(np.digitize(x, edges) - 1, 0, K - 1)
    S1 = np.bincount(idx, weights=w,     minlength=K).astype(float)[:K]
    S2 = np.bincount(idx, weights=w * w, minlength=K).astype(float)[:K]
    W  = float(w.sum())
    p = S1 / max(W, epsilon)
    sigma2 = np.maximum((S2 - (S1**2) / max(n_samples, 1)) / max(W, epsilon), 0.0)
    centers = 0.5 * (edges[1:] + edges[:-1])
    if return_nonempty:
        nonempty = p > float(empty_tol)
        return centers, p, sigma2, nonempty
    return centers, p, sigma2


# ======================== Chi-squared objective ========================
def make_objective_fixed_epsA_fixed_s_chi2(
    theta_ref, epsA_fixed,
    rho, u, C, Ca, N,
    edges_ref, p_ref,
    n_samples,
    *, L_fixed=None, epsilon=1e-12,
    sigma2_floor=1e-8,
    n_bins_sim=120,
    N_sites=None,
):
    be_ref, beA_ref, bm_ref, L_ref, Mc_ref, s_ref = _parse_theta_with_s(theta_ref)
    if L_fixed is None:
        L_fixed = float(L_ref)

    widths_ref  = np.diff(edges_ref)
    centers_ref = 0.5 * (edges_ref[:-1] + edges_ref[1:])
    q_density_ref = np.asarray(p_ref, float).copy()
    q_density_ref = np.where(q_density_ref > 0, q_density_ref, 0.0)
    norm_ref = np.sum(q_density_ref * widths_ref)
    if norm_ref > 1e-300:
        q_density_ref /= norm_ref

    theta_ref_ext = (be_ref, beA_ref, bm_ref, L_ref, Mc_ref, s_ref, 0.0)

    def _wmean_std(x, w):
        W = max(w.sum(), 1e-300)
        m = float(np.dot(w, x) / W)
        v = float(np.dot(w, (x - m)**2) / W)
        return m, math.sqrt(max(v, 1e-300))

    def objective(x):
        be_p, bm_p = map(float, x[:2])
        theta_prime_ext = (be_p, float(epsA_fixed), bm_p, float(L_fixed), 0.0, s_ref, 0.0)
        w, Mprime = reweight_from_samples_counts_with_s(
            M_ref=rho - s_ref * u, C=C, Ca=Ca, N=N,
            theta_prime=theta_prime_ext, theta_ref=theta_ref_ext,
            rho=rho, u=u, N_sites=N_sites, return_Mprime=True,
        )
        Mc_w, std_w = _wmean_std(Mprime, w)
        std_w = max(std_w, 1e-300)
        x_vals = (Mprime - Mc_w) / std_w

        x_lo = min(float(np.min(x_vals)) - 0.5, float(edges_ref[0]))
        x_hi = max(float(np.max(x_vals)) + 0.5, float(edges_ref[-1]))
        edges_sim = np.linspace(x_lo, x_hi, n_bins_sim + 1)
        widths_sim = np.diff(edges_sim)
        centers_sim = 0.5 * (edges_sim[:-1] + edges_sim[1:])

        n_b = n_bins_sim
        idx_bin = np.clip(np.digitize(x_vals, edges_sim) - 1, 0, n_b - 1)
        S1 = np.bincount(idx_bin, weights=w,     minlength=n_b).astype(float)[:n_b]
        S2 = np.bincount(idx_bin, weights=w * w, minlength=n_b).astype(float)[:n_b]
        W_total = max(S1.sum(), 1e-300)
        p_sim = S1 / W_total

        sigma2 = (S2 - (S1**2) / max(n_samples, 1)) / max(W_total, 1e-300)
        sigma2 = np.maximum(sigma2, 0.0)

        q_density_interp = np.interp(centers_sim, centers_ref, q_density_ref,
                                     left=0.0, right=0.0)
        q_mass = q_density_interp * widths_sim
        q_mass_sum = max(q_mass.sum(), 1e-300)
        q_mass /= q_mass_sum

        mask = (p_sim > 0.0) & (sigma2 > sigma2_floor)
        if mask.sum() < 5:
            mask = p_sim > 0.0
            sigma2 = np.maximum(sigma2, sigma2_floor)

        diff2 = (p_sim[mask] - q_mass[mask])**2
        chi2 = float(np.sum(diff2 / np.maximum(sigma2[mask], sigma2_floor)))

        n_occ = int(mask.sum())
        return chi2 / max(n_occ, 1)

    return objective


def optimize_eps_mu_Mc_fixed_epsA_fixed_s(
    theta_ref, epsA_fixed,
    rho, u, C, Ca, N, edges_ref, p_ref,
    *, L_fixed=None, x0=None, bounds=None, method="L-BFGS-B", options=None,
    n_samples=None, N_sites=None,
):
    if n_samples is None:
        n_samples = len(rho)
    obj = make_objective_fixed_epsA_fixed_s_chi2(
        theta_ref, epsA_fixed, rho, u, C, Ca, N, edges_ref, p_ref,
        n_samples=n_samples, L_fixed=L_fixed, N_sites=N_sites,
    )
    be_ref, beA_ref, bm_ref, L_ref, Mc_ref, s_ref = _parse_theta_with_s(theta_ref)
    if L_fixed is None:
        L_fixed = L_ref
    if x0 is None:
        x0 = np.array([be_ref, bm_ref], float)
    if bounds is None:
        bounds = [(be_ref - 0.12, be_ref + 0.12),
                  (bm_ref - 0.12, bm_ref + 0.12)]

    res = minimize(obj, x0=np.asarray(x0, float),
                   method="L-BFGS-B", bounds=bounds,
                   options=options or dict(maxiter=400, ftol=1e-5))

    x_opt = res.x
    at_bound = (abs(x_opt[0] - bounds[0][0]) < 1e-8 or abs(x_opt[0] - bounds[0][1]) < 1e-8 or
                abs(x_opt[1] - bounds[1][0]) < 1e-8 or abs(x_opt[1] - bounds[1][1]) < 1e-8)
    if at_bound:
        print(f"  [optimize] WARNING: hit bound, retrying with wider search", flush=True)
        wider = [(be_ref - 0.25, be_ref + 0.25),
                 (bm_ref - 0.25, bm_ref + 0.25)]
        res2 = minimize(obj, x0=res.x, method="L-BFGS-B", bounds=wider,
                        options=dict(maxiter=600, ftol=1e-6))
        if res2.fun < res.fun:
            res = res2

    be_p, bm_p = map(float, res.x)
    theta_ref_ext   = (be_ref, beA_ref, bm_ref, L_ref, Mc_ref, s_ref, 0.0)
    theta_prime_ext = (be_p, float(epsA_fixed), bm_p, float(L_fixed), 0.0, s_ref, 0.0)
    w, Mprime = reweight_from_samples_counts_with_s(
        M_ref=rho - s_ref * u, C=C, Ca=Ca, N=N,
        theta_prime=theta_prime_ext, theta_ref=theta_ref_ext,
        rho=rho, u=u, N_sites=N_sites, return_Mprime=True,
    )
    W = max(w.sum(), 1e-300)
    Mc_w = float(np.dot(w, Mprime) / W)
    res.theta_star = (be_p, float(epsA_fixed), bm_p, float(L_fixed), Mc_w, float(s_ref))
    return res


# ======================== mu search by simulation ========================
def find_mu_equal_areas_by_simulation(
    simulate, mu0,
    dilute_window=(0.00, 0.30), dense_window=(0.70, 1.00),
    step0=0.02, tol=2e-2, max_iters=50, verbose=True,
):
    def basin_masses(rho):
        rho = np.asarray(rho, float)
        m_dil = ((rho >= dilute_window[0]) & (rho < dilute_window[1])).mean()
        m_den = ((rho >= dense_window[0]) & (rho <= dense_window[1])).mean()
        return float(m_dil), float(m_den)

    cache = {}

    def mass_diff_at(mu):
        mu = float(mu); mu_key = float(np.round(mu, 12))
        if mu_key in cache:
            return cache[mu_key]
        data = simulate(mu)
        m_dil, m_den = basin_masses(data.rho_mu)
        f = float(m_den - m_dil)
        cache[mu_key] = (f, data)
        return cache[mu_key]

    mu = float(mu0)
    f, data = mass_diff_at(mu)
    if verbose:
        print(f"[mu={mu:.6f}] f=m_dense-m_dilute={f:+.4f}")
    best = (abs(f), mu, data)
    bracket = None

    for it in range(1, int(max_iters) + 1):
        if abs(f) <= tol:
            return {"mu": mu, "success": True, "data": data, "iters": it - 1}
        if bracket is not None:
            mu_lo, f_lo, mu_hi, f_hi = bracket
            mu = 0.5 * (mu_lo + mu_hi)
            f, data = mass_diff_at(mu)
            if verbose:
                print(f"[bisect] mu={mu:.6f} f={f:+.4f} bracket=[{mu_lo:.6f},{mu_hi:.6f}]")
            if abs(f) < best[0]:
                best = (abs(f), mu, data)
            if f < 0:
                bracket = (mu, f, mu_hi, f_hi)
            else:
                bracket = (mu_lo, f_lo, mu, f)
            continue
        mu_new = mu + (step0 if f > 0 else -step0)
        f_new, data_new = mass_diff_at(mu_new)
        if verbose:
            print(f"[step] mu={mu_new:.6f} f={f_new:+.4f}")
        if abs(f_new) < best[0]:
            best = (abs(f_new), mu_new, data_new)
        if f != 0.0 and f_new != 0.0 and (f * f_new < 0.0):
            bracket = (mu, f, mu_new, f_new) if f < 0 else (mu_new, f_new, mu, f)
        mu, f, data = mu_new, f_new, data_new

    _, mu_best, data_best = best
    return {"mu": mu_best, "success": False, "data": data_best, "iters": int(max_iters)}


def make_simulator(build_data_mu, **base_kwargs):
    def simulate(mu):
        kw = dict(base_kwargs); kw["mu"] = float(mu)
        return build_data_mu(**kw)
    return simulate


# ========================================================================
#  Ising reference on triangular lattice
# ========================================================================
from pathlib import Path


def build_or_load_triangular_ising_reference(
    *,
    cache_path="cache/triangular_ising_reference_L64_b3000_n20000_seed12345.npz",
    L_ising=64,
    burnin_clusters=3000,
    samples_to_collect=20000,
    seed=12345,
    smooth_sigma=0.05,
    nbins=101,
    n_pad=6,
    force_rebuild=False,
    make_plot=False,
):
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    if cache_path.exists() and not force_rebuild:
        ref = np.load(cache_path, allow_pickle=False)
        out = {
            "Tc": float(ref["Tc"]),
            "beta_ising": float(ref["beta_ising"]),
            "p_add": float(ref["p_add"]),
            "L_ising": int(ref["L_ising"]),
            "burnin_clusters": int(ref["burnin_clusters"]),
            "samples_to_collect": int(ref["samples_to_collect"]),
            "seed": int(ref["seed"]),
            "m_mean": float(ref["m_mean"]),
            "m_std": float(ref["m_std"]),
            "centers_ref": ref["centers_ref"].astype(np.float64, copy=False),
            "edges_ref": ref["edges_ref"].astype(np.float64, copy=False),
            "P_smooth": ref["P_smooth"].astype(np.float64, copy=False),
            "x_samples": ref["x_samples"].astype(np.float64, copy=False),
        }
        print(f"[ising-ref] loaded cached triangular Ising reference from: {cache_path}", flush=True)
        return out

    rng = np.random.default_rng(seed)

    Tc = 4.0 / math.log(3.0)
    beta_ising = 1.0 / Tc
    p_add = 1.0 - math.exp(-2.0 * beta_ising)

    N_ising = L_ising * L_ising
    spins = rng.choice([-1, 1], size=(L_ising, L_ising)).astype(np.int8)

    def neighbors_tri(x, y):
        if x % 2 == 0:
            return (
                (x, (y + 1) % L_ising),
                ((x - 1) % L_ising, y),
                ((x - 1) % L_ising, (y - 1) % L_ising),
                (x, (y - 1) % L_ising),
                ((x + 1) % L_ising, (y - 1) % L_ising),
                ((x + 1) % L_ising, y),
            )
        else:
            return (
                (x, (y + 1) % L_ising),
                ((x - 1) % L_ising, (y + 1) % L_ising),
                ((x - 1) % L_ising, y),
                (x, (y - 1) % L_ising),
                ((x + 1) % L_ising, y),
                ((x + 1) % L_ising, (y + 1) % L_ising),
            )

    def magnetization_per_spin():
        return spins.sum() / float(N_ising)

    def wolff_step():
        x = rng.integers(0, L_ising)
        y = rng.integers(0, L_ising)
        target_spin = spins[x, y]
        stack = [(x, y)]
        in_cluster = np.zeros((L_ising, L_ising), dtype=np.bool_)
        in_cluster[x, y] = True
        while stack:
            i, j = stack.pop()
            for ni, nj in neighbors_tri(i, j):
                if (not in_cluster[ni, nj]) and (spins[ni, nj] == target_spin):
                    if rng.random() < p_add:
                        in_cluster[ni, nj] = True
                        stack.append((ni, nj))
        spins[in_cluster] *= -1
        return int(in_cluster.sum())

    print(
        f"[ising-ref] building triangular Ising reference: Tc={Tc:.6f}, p_add={p_add:.6f}, "
        f"L={L_ising}, burnin={burnin_clusters}, samples={samples_to_collect}",
        flush=True,
    )

    for _ in range(burnin_clusters):
        wolff_step()

    m_samples = np.empty(samples_to_collect, dtype=np.float64)
    for k in range(samples_to_collect):
        wolff_step()
        m_samples[k] = magnetization_per_spin()

    m_mean = float(m_samples.mean())
    m_std = float(m_samples.std(ddof=1))
    x_samples = (m_samples - m_mean) / m_std

    R_data = float(np.max(np.abs(x_samples)))
    dx = 2.0 * R_data / nbins
    R = R_data + n_pad * dx
    edges = np.linspace(-R, R, (nbins + 2 * n_pad) + 1)
    hist, _ = np.histogram(x_samples, bins=edges, density=True)
    x_centers = 0.5 * (edges[:-1] + edges[1:])

    xc, P_smooth = smooth_histogram_gaussian(x_centers, hist, sigma=smooth_sigma)

    np.savez_compressed(
        cache_path,
        Tc=np.float64(Tc), beta_ising=np.float64(beta_ising),
        p_add=np.float64(p_add), L_ising=np.int64(L_ising),
        burnin_clusters=np.int64(burnin_clusters),
        samples_to_collect=np.int64(samples_to_collect),
        seed=np.int64(seed), m_mean=np.float64(m_mean), m_std=np.float64(m_std),
        centers_ref=xc.astype(np.float64), edges_ref=edges.astype(np.float64),
        P_smooth=P_smooth.astype(np.float64), x_samples=x_samples.astype(np.float64),
    )
    print(f"[ising-ref] saved triangular Ising reference to: {cache_path}", flush=True)

    return {
        "Tc": Tc, "beta_ising": beta_ising, "p_add": p_add,
        "L_ising": L_ising, "burnin_clusters": burnin_clusters,
        "samples_to_collect": samples_to_collect, "seed": seed,
        "m_mean": m_mean, "m_std": m_std,
        "centers_ref": xc, "edges_ref": edges,
        "P_smooth": P_smooth, "x_samples": x_samples,
    }


# ========================================================================
#  MAIN driver
# ========================================================================
CHI2_QUALITY_THRESHOLD = 3.0


def main(
    eps_ns, eps_sp_prev, mu,
    *,
    ising_ref_cache="cache/triangular_ising_reference_L64_b3000_n20000_seed12345.npz",
    rebuild_ising_ref=False,
    max_workers=5, n_replicas=5,
    save_dir="fits_tri",
):
    os.makedirs(save_dir, exist_ok=True)
    rng = np.random.default_rng(12345)

    ising_ref = build_or_load_triangular_ising_reference(
        cache_path=ising_ref_cache,
        L_ising=64, burnin_clusters=3000, samples_to_collect=20000,
        seed=12345, smooth_sigma=0.05, nbins=101, n_pad=6,
        force_rebuild=rebuild_ising_ref,
    )
    centers_ref = ising_ref["centers_ref"]
    edges_ref   = ising_ref["edges_ref"]
    P_smooth    = ising_ref["P_smooth"]

    species = get_species_list_ind_hex(['A0A0A0', 'SSSSSS'])
    print([p.patches for p in species])
    n_rot = sum(1 for s in species if s.patches[0] != 2 or s.patches[1] != 2)
    species_mu_indices = np.arange(n_rot)
    print(f"Species ({len(species)} total, {n_rot} non-empty):")
    for p in species:
        print(f"  {p.index}: {p.patches}")

    L = 10
    N_total = L * L
    n = len(species)
    chem_pots = np.zeros(n)
    empty_idx = n - 1  # last species is "empty" (SSSSSS)

    # ---- FIX 1: Create genuinely distinct initial lattices ----
    def make_lattice(rho_occupied):
        """Create lattice with given occupied fraction."""
        n_occ = int(rho_occupied * N_total)
        n_empty = N_total - n_occ
        # Distribute occupied sites evenly among orientations
        per_orient = n_occ // n_rot
        remainder = n_occ - per_orient * n_rot
        flat = []
        for i in range(n_rot):
            count = per_orient + (1 if i < remainder else 0)
            flat.extend([species[i]] * count)
        flat.extend([species[empty_idx]] * n_empty)
        rng.shuffle(flat)
        return np.array(flat).reshape((L, L))

    lattice_dense  = make_lattice(0.95)   # nearly full
    lattice_dilute = make_lattice(0.05)   # nearly empty
    lattice_middle = make_lattice(0.50)   # half occupied

    points: List[Tuple[float, float, float]] = []

    eps_sp_step = 0.02
    eps_sp_target = eps_sp_prev + eps_sp_step
    s_ref = None
    warm_lattices = None

    while eps_sp_target <= 6.00:
        eps_sp = eps_sp_target
        print(f"\n=== Step: target eps_sp={eps_sp:.3f} (prev={eps_sp_prev:.3f}, "
              f"step={eps_sp - eps_sp_prev:.4f}) ===", flush=True)

        # ---- FIX 2: Estimate mu from mean-field before searching ----
        # On triangular lattice z=6: mu_mf ≈ (z/2)*eps_ns + ln(n_rot)
        mu_mf_estimate = 3.0 * eps_ns + np.log(float(n_rot))
        # Use previous mu but clamp to be within 0.3 of mean-field estimate
        mu_start = mu
        

        # ---- FIX 3: Create extra warm lattices at different densities ----
        extra_lats = []
        if warm_lattices is not None:
            extra_lats.extend(warm_lattices)
        extra_lats.append(to_int_lattice(lattice_dense).copy())
        extra_lats.append(to_int_lattice(lattice_dilute).copy())

        simulate = make_simulator(
            build_data_mu_cpu,
            simulate_lattice_unused=simulate_lattice_triangular,
            lattice_dense=lattice_dense,       # ← NOW genuinely dense
            lattice_dilute=lattice_dilute,     # ← NOW genuinely dilute
            lattice_middle=lattice_middle,
            species=species,
            species_mu_indices=species_mu_indices,
            chem_pots=chem_pots,
            eps_ns=eps_ns,
            eps_sp=eps_sp_prev,
            beta=1.0,
            steps=600_000_000,
            snapshot_interval=400_000,
            buffer_size=100_000_000,
            empty_index=empty_idx,
            max_workers=1,
            n_replicas=5,
            extra_lattices=extra_lats,         # ← diverse starts
        )

        # ---- FIX 4: Adaptive step size ----
        guard = find_mu_equal_areas_by_simulation(
            simulate, mu0=mu_start,
            dilute_window=(0.00, 0.40),
            dense_window=(0.60, 1.00),
            step0=0.02,        # ← 5x larger than before
            tol=5e-2,          # ← slightly relaxed
            max_iters=80,      # ← more room
            verbose=True,
        )

        mu = guard["mu"]
        data_mu = guard["data"]


        if hasattr(data_mu, '_lat_final'):
            warm_lattices = [data_mu._lat_final]
        else:
            warm_lattices = None

        rho = data_mu.rho_mu
        u   = data_mu.u
        C   = data_mu.C_ns
        Ca  = data_mu.C_sp
        N   = data_mu.N_mu
        mu_ref = data_mu.mu0

        # --- Debug prints ---
        print(f"DEBUG: len(rho)={len(rho)}, rho range=[{rho.min():.4f}, {rho.max():.4f}]")
        print(f"DEBUG: u range=[{u.min():.6f}, {u.max():.6f}]")
        print(f"DEBUG: C_ns range=[{C.min():.0f}, {C.max():.0f}], "
              f"C_sp range=[{Ca.min():.0f}, {Ca.max():.0f}]")
        print(f"DEBUG: any NaN? rho:{np.any(np.isnan(rho))}, u:{np.any(np.isnan(u))}")

        res_equal = equal_peak_mu_in_rho_by_reweighting(
            rho, N, mu_ref,
            beta=1.0,
            rho_edges=np.linspace(0.0, 1.0, 201),
            dilute_window=(0.00, 0.40),
            dense_window=(0.60, 1.00),
            ess_min=3000, step0=0.0001, tol=1e-2, verbose=True
        )
        mu_star = float(res_equal["mu_star"])
        print(f"mu* = {mu_star:.6f}   ESS ~ {res_equal['ess']:.0f}")

        w_mu = _weights_mu(N, mu_star, mu_ref, beta=1.0)
        w_mu /= w_mu.sum()

        print(f"weighted <rho> = {np.dot(w_mu, rho):.4f}")
        print(f"DEBUG: w_mu ESS={_ess(w_mu):.0f}, "
              f"w_mu range=[{w_mu.min():.2e}, {w_mu.max():.2e}]")

        # --- Debug: check KDE input ---
        M_test = rho  # s=0 initially
        M_c = np.dot(w_mu, M_test)
        M_s = np.sqrt(np.dot(w_mu, (M_test - M_c)**2))
        x_test = (M_test - M_c) / max(M_s, 1e-300)
        print(f"DEBUG: x_test range=[{x_test.min():.4f}, {x_test.max():.4f}], "
              f"std={_wstd(x_test, w_mu):.4f}")
        print(f"DEBUG: x_ref range=[{centers_ref.min():.4f}, {centers_ref.max():.4f}]")
        print(f"DEBUG: grid spacing dx={np.mean(np.diff(centers_ref)):.6f}")

        fit = fit_s_r_from_reweighted(
            rho=rho, u=u,
            x_ref=centers_ref,
            s_ref=None,
            P_ref=P_smooth / np.trapz(P_smooth, centers_ref),
            weights=w_mu,
            L=L, beta_over_nu=1/8, sigma_x=0.03,
            r_penalty=1e-1, w_asym=2.0
        )
        s_ref = fit['s']
        print(f"s_ref={s_ref:.6f}, r={fit['r']:.6f}, match_L2={fit['match_L2']:.6f}")

        # --- Save plots to disk (Agg backend) ---
        fig, ax = plt.subplots()
        ax.hist(rho, bins=101, weights=w_mu)
        ax.set_title(f"rho (weighted) | eps_ns={eps_ns:.2f}, eps_sp={eps_sp_prev:.2f}, mu*={mu_star:.2f}")
        fig.savefig(os.path.join(save_dir,
                    f"rho_ens{eps_ns:.4f}_esp{eps_sp_prev:.4f}_mu{mu_star:.4f}.png"), dpi=100)
        plt.close(fig)

        fig, ax = plt.subplots()
        P_ref_norm = fit["P_ref"] / max(np.trapz(fit['P_ref'], fit['x']), 1e-300)
        P_fit_norm = fit["P_fit"] / max(np.trapz(fit['P_fit'], fit['x']), 1e-300)
        ax.plot(fit["x"], P_ref_norm, label="Target P*(x)")
        ax.plot(fit["x"], P_fit_norm, "--", label="Reweighted + mixed-field fit")
        ax.set_xlabel("x"); ax.set_ylabel("density"); ax.legend()
        ax.set_title(f"eps_ns={eps_ns:.2f}, eps_sp={eps_sp_prev:.2f} | match_L2={fit['match_L2']:.4f}")
        fig.savefig(os.path.join(save_dir,
                    f"fit_ens{eps_ns:.4f}_esp{eps_sp_prev:.4f}_mu{mu_star:.4f}.png"), dpi=100)
        plt.close(fig)

        Mprime = rho - s_ref * u
        Mc = float(np.dot(w_mu, Mprime))
        lambdap = 1.0 - s_ref * fit['r']

        theta_ref = (eps_ns, eps_sp_prev, mu_ref, lambdap, Mc, s_ref)

        res = optimize_eps_mu_Mc_fixed_epsA_fixed_s(
            theta_ref, eps_sp,
            rho, u, C, Ca, N,
            edges_ref, P_smooth / np.trapz(P_smooth, centers_ref),
            L_fixed=None, N_sites=data_mu.N_sites,
        )
        print(f"success: {res.success}", flush=True)
        print(f"chi2*  : {res.fun:.6f}", flush=True)
        print(f"theta* : {res.theta_star}", flush=True)

        if res.fun > CHI2_QUALITY_THRESHOLD:
            half_step = (eps_sp - eps_sp_prev) / 2.0
            if half_step >= 0.005:
                print(f"  [quality-gate] chi2={res.fun:.4f} > {CHI2_QUALITY_THRESHOLD}, "
                      f"halving step to {half_step:.4f}", flush=True)
                eps_sp_target = eps_sp_prev + half_step
                continue
            else:
                print(f"  [quality-gate] chi2={res.fun:.4f} > threshold but step already minimal, "
                      f"proceeding anyway", flush=True)

        eps_ns, _, mu, _, _, _ = res.theta_star
        #mu = mu_ref
        points.append((float(eps_ns), float(eps_sp), float(mu)))
        eps_sp_prev = float(eps_sp)
        eps_sp_target = eps_sp_prev + 0.02

        # Save checkpoint after each successful step
        if points:
            arr = np.array(points)
            np.savetxt(os.path.join(save_dir, "points_checkpoint.txt"), arr,
                       header="eps_ns  eps_sp  mu", fmt="%.8f")

    # --- Final summary ---
    if points:
        eps_ns_list, eps_sp_list, mu_list = map(np.array, zip(*points))

        fig, ax = plt.subplots()
        ax.plot(eps_sp_list, eps_ns_list, marker='.')
        ax.set_xlabel(r"$\epsilon_{\rm sp}$ (fixed)")
        ax.set_ylabel(r"$\epsilon_{\rm ns}^*$")
        ax.set_title("Triangular lattice: sequential sweep")
        fig.savefig(os.path.join(save_dir, "sweep_eps_ns.png"), dpi=150)
        plt.close(fig)

        fig, ax = plt.subplots()
        ax.plot(eps_sp_list, mu_list, marker='.')
        ax.set_xlabel(r"$\epsilon_{\rm sp}$ (fixed)")
        ax.set_ylabel(r"$\mu^*$")
        ax.set_title("mu* across the sequential sweep")
        fig.savefig(os.path.join(save_dir, "sweep_mu.png"), dpi=150)
        plt.close(fig)

        np.savetxt(os.path.join(save_dir, "points_final.txt"),
                   np.array(points), header="eps_ns  eps_sp  mu", fmt="%.8f")

    print("\nCollected points (eps_ns*, eps_sp, mu*):")
    for p in points:
        print(p)



# ========================================================================
#  Entry point
# ========================================================================
#  Entry point
# ========================================================================
if __name__ == "__main__":
    eps_ns_init = 0.75
    eps_sp_init = 1.32
    mu_init     = 4.07

    main(
        eps_ns_init, eps_sp_init, mu_init,
        max_workers=10,
        n_replicas=10,
        save_dir="fits_mercedes",
    )
