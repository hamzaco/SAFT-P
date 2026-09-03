"""
Reduced SAFT-P solver for n x n square plaquettes.

This is the solver used in ``spinodal_stick_shaped.ipynb`` /
``spinodal_l_shaped.ipynb`` with the hard-coded ``4`` (number of sites in a 2x2
plaquette) promoted to a parameter ``nsite = n*n``.  Everything else is
unchanged:

*  unknowns are the scalar composition multiplier ``mu`` and the free
   super-patch abundances ``W`` (one per boundary super-edge type);
*  the class populations ``rho_p`` are eliminated analytically by a softmax,
   with the implicit derivative correction of the association term;
*  residuals are ``A^T rho - phi`` and the mass-action conditions
   ``W_s = X_s * (S^T rho)_s``.

The number of super-edge types (and hence the Newton dimension) is 9 for 2x2
and 27 for 3x3 with a binary patch alphabet plus a vacancy patch; the number of
plaquette classes P enters only through the softmax and the P x F matrix S.

Two parameterisations of the mass-action block are available:
  ``logW=True``  (default) solves in ``y = log W``, which keeps W strictly
                 positive.  With 27 super-patch types this matters: the direct
                 W Newton step occasionally lands on a nonphysical branch.
  ``logW=False`` reproduces the original direct-W iteration exactly.

Both converge to the same root; only the path differs.
"""

from __future__ import annotations


import numpy as np

try:
    from numba import njit as _njit, prange
    _HAS_NUMBA = True
except ImportError:                                            # pragma: no cover
    _HAS_NUMBA = False
    prange = range

    def _njit(*args, **kwargs):
        def _wrap(fn):
            return fn
        if args and callable(args[0]):
            return args[0]
        return _wrap


# ---------------------------------------------------------------------------
# smoothing / envelope helpers  (numpy ports of the notebook versions)
# ---------------------------------------------------------------------------

def loess_derivs(x, y, *, bandwidth=0.15, order=2):
    """Local quadratic regression; returns (f, f', f'').  Same weights, knots
    and ridge as the torch version in the published notebooks."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n = x.shape[0]
    f0 = np.zeros(n); f1 = np.zeros(n); f2 = np.zeros(n)

    h = bandwidth * (x.max() - x.min())
    if h <= 0:
        h = 1.0

    for i in range(n):
        d = np.abs(x - x[i])
        w = np.clip(1.0 - (d / h) ** 3, 0.0, None) ** 3
        dx = x - x[i]
        cols = [np.ones_like(dx)]
        if order >= 1:
            cols.append(dx)
        if order >= 2:
            cols.append(dx ** 2)
        X = np.stack(cols, axis=1)
        XTW = X.T * w
        M = XTW @ X
        beta = np.linalg.solve(M + 1e-12 * np.eye(M.shape[0]), XTW @ y)
        f0[i] = beta[0]
        f1[i] = beta[1] if order >= 1 else np.nan
        f2[i] = 2.0 * beta[2] if order >= 2 else np.nan
    return f0, f1, f2


def zero_crossings_linear(phis, y, tol=1e-6):
    z = []
    for i in range(len(phis) - 1):
        y1, y2 = y[i], y[i + 1]
        if abs(y1) < tol:
            z.append(float(phis[i]))
            continue
        if y1 * y2 < 0:
            t = -y1 / (y2 - y1)
            z.append(float(phis[i] + t * (phis[i + 1] - phis[i])))
    if abs(y[-1]) < tol:
        z.append(float(phis[-1]))
    return z


def extract_binodals_from_convex_envelope(phi, f, *, coexist_tol=1e-5, min_gap_points=6):
    from scipy.spatial import ConvexHull
    phi = np.asarray(phi, float)
    f = np.asarray(f, float)
    pts = np.column_stack([phi, f])
    try:
        hull = ConvexHull(pts)
    except Exception:
        return {"phi": phi, "f": f, "hull_idx": [], "segments": []}

    hull_idx = np.sort(hull.vertices)
    out = []
    for k in range(len(hull_idx) - 1):
        i, j = hull_idx[k], hull_idx[k + 1]
        if j - i < min_gap_points:
            continue
        phi1, phi2 = phi[i], phi[j]
        f1, f2 = f[i], f[j]
        mu_line = (f2 - f1) / (phi2 - phi1 + 1e-30)
        line = f1 + mu_line * (phi - phi1)
        diff = f - line
        mask = (phi >= phi1) & (phi <= phi2)
        if not np.any(mask):
            continue
        barrier = float(np.max(diff[mask]))
        if barrier < coexist_tol:
            continue
        out.append({"phi1": float(phi1), "phi2": float(phi2), "mu": float(mu_line),
                    "barrier": barrier, "below_min": float(np.min(diff[mask])),
                    "n_points": int(j - i)})
    return {"phi": phi, "f": f, "hull_idx": hull_idx.tolist(), "segments": out}


def best_binodal_segment(result, *, prefer="largest_barrier"):
    segs = result.get("segments", [])
    if not segs:
        return None
    if prefer == "widest":
        key = lambda d: (d["phi2"] - d["phi1"], d["barrier"])     # noqa: E731
    elif prefer == "most_points":
        key = lambda d: (d["n_points"], d["barrier"])             # noqa: E731
    else:
        key = lambda d: (d["barrier"], d["phi2"] - d["phi1"])     # noqa: E731
    return max(segs, key=key)


def common_tangent_line(phi, f, binodal):
    phi = np.asarray(phi, float); f = np.asarray(f, float)
    phi1 = float(binodal["phi1"]); mu = float(binodal["mu"])
    f1 = float(np.interp(phi1, phi, f))
    c = f1 - mu * phi1
    return mu * phi + c, c


def excess_above_tangent(phi, f, binodal):
    line, _ = common_tangent_line(phi, f, binodal)
    return f - line, line


# ---------------------------------------------------------------------------
# reduced residual and Newton solve
# ---------------------------------------------------------------------------

def build_S_matrix(patch_to_species, patch_to_small, m_patch, P, F):
    S = np.zeros((P, F), dtype=np.float64)
    for p in range(len(patch_to_species)):
        S[patch_to_species[p], patch_to_small[p]] += m_patch[p]
    return np.ascontiguousarray(S)


@_njit(cache=True)
def _eval_residual_reduced(mu, W, phi, S, delta, A, e, lm, nsite):
    Fd = W.shape[0]
    P = A.shape[0]

    u = delta @ W
    X = np.empty(Fd)
    hv = np.empty(Fd)
    for s in range(Fd):
        xs = 1.0 / max(1.0 + u[s], 1e-12)
        xs = max(min(xs, 1.0), 1e-12)
        X[s] = xs
        hv[s] = np.log(max(xs, 1e-300)) - 0.5 * xs + 0.5

    # first pass: envelope term only
    g0 = S @ hv
    lr = np.empty(P)
    mx = -1e300
    for i in range(P):
        v = nsite * mu * A[i] - e[i] - g0[i] + lm[i]
        lr[i] = v
        if v > mx:
            mx = v
    Z = 0.0
    rho = np.empty(P)
    for i in range(P):
        v = np.exp(lr[i] - mx)
        rho[i] = v
        Z += v
    inv_Z = 1.0 / Z
    for i in range(P):
        rho[i] *= inv_Z

    pr = np.zeros(Fd)
    for i in range(P):
        ri = rho[i]
        for s in range(Fd):
            pr[s] += S[i, s] * ri

    # implicit derivative correction
    K = np.empty((Fd, Fd))
    for s in range(Fd):
        xs2 = X[s] * X[s]
        for sp in range(Fd):
            K[s, sp] = xs2 * delta[s, sp] * pr[sp]

    dAdX = np.empty(Fd)
    for s in range(Fd):
        dAdX[s] = pr[s] * (1.0 / max(X[s], 1e-12) - 0.5)

    Msys = np.empty((Fd, Fd))
    for s in range(Fd):
        for sp in range(Fd):
            Msys[s, sp] = K[sp, s]
        Msys[s, s] += 1.0 + 1e-12
    alpha = np.linalg.solve(Msys, dAdX)

    beta = np.zeros(Fd)
    for s in range(Fd):
        acc = 0.0
        for sp in range(Fd):
            acc += delta[sp, s] * alpha[sp] * X[sp] * X[sp]
        beta[s] = -acc

    hv_corr = np.empty(Fd)
    for s in range(Fd):
        hv_corr[s] = hv[s] + beta[s] * X[s]

    g = S @ hv_corr
    mx = -1e300
    for i in range(P):
        v = nsite * mu * A[i] - e[i] - g[i] + lm[i]
        lr[i] = v
        if v > mx:
            mx = v
    Z = 0.0
    for i in range(P):
        v = np.exp(lr[i] - mx)
        rho[i] = v
        Z += v
    inv_Z = 1.0 / Z
    for i in range(P):
        rho[i] *= inv_Z

    pr[:] = 0.0
    for i in range(P):
        ri = rho[i]
        for s in range(Fd):
            pr[s] += S[i, s] * ri

    R = np.empty(1 + Fd)
    phi_rho = 0.0
    for i in range(P):
        phi_rho += A[i] * rho[i]
    R[0] = phi_rho - phi
    for s in range(Fd):
        R[1 + s] = W[s] - X[s] * pr[s]
    return R, rho, X, pr


@_njit(cache=True)
def _eval_residual_reduced_logW(mu, y, phi, S, delta, A, e, lm, nsite):
    Fd = y.shape[0]
    W = np.empty(Fd)
    for s in range(Fd):
        ys = max(min(y[s], 30.0), -50.0)
        W[s] = np.exp(ys)
    R_lin, rho, X, pr = _eval_residual_reduced(mu, W, phi, S, delta, A, e, lm, nsite)
    R = np.empty(1 + Fd)
    R[0] = R_lin[0]
    for s in range(Fd):
        R[1 + s] = y[s] - np.log(max(X[s] * pr[s], 1e-300))
    return R, rho, X, pr, W


@_njit(cache=True)
def _find_initial_mu(phi_target, A, lm, e, S, delta, W, nsite):
    Fd = W.shape[0]
    P = A.shape[0]
    u = delta @ W
    hv = np.empty(Fd)
    for s in range(Fd):
        xs = 1.0 / max(1.0 + u[s], 1e-12)
        xs = max(min(xs, 1.0), 1e-12)
        hv[s] = np.log(max(xs, 1e-300)) - 0.5 * xs + 0.5
    g = S @ hv

    mu_lo = -100.0
    mu_hi = 100.0
    mu = 0.0
    for _ in range(120):
        mx = -1e300
        for i in range(P):
            v = nsite * mu * A[i] - e[i] - g[i] + lm[i]
            if v > mx:
                mx = v
        Z = 0.0
        for i in range(P):
            Z += np.exp(nsite * mu * A[i] - e[i] - g[i] + lm[i] - mx)
        inv_Z = 1.0 / Z
        phi_mu = 0.0
        dphi = 0.0
        for i in range(P):
            ri = np.exp(nsite * mu * A[i] - e[i] - g[i] + lm[i] - mx) * inv_Z
            phi_mu += A[i] * ri
            dphi += nsite * A[i] * A[i] * ri
        dphi -= nsite * phi_mu * phi_mu

        err = phi_mu - phi_target
        if abs(err) < 1e-12:
            break
        if err > 0.0:
            mu_hi = mu
        else:
            mu_lo = mu
        if abs(dphi) > 1e-30:
            mu_new = mu - err / dphi
            if mu_new < mu_lo or mu_new > mu_hi:
                mu_new = 0.5 * (mu_lo + mu_hi)
        else:
            mu_new = 0.5 * (mu_lo + mu_hi)
        mu = mu_new
    return mu


@_njit(cache=True)
def _free_energy_from_state(rho, X, pr, A, e, lm, nsite):
    Fd = X.shape[0]
    P = A.shape[0]
    A_mix = 0.0
    for i in range(P):
        A_mix += rho[i] * (np.log(max(rho[i], 1e-300)) - lm[i])
    A_lin = 0.0
    for i in range(P):
        A_lin += e[i] * rho[i]
    A_assoc = 0.0
    for s in range(Fd):
        hv = np.log(max(X[s], 1e-300)) - 0.5 * X[s] + 0.5
        A_assoc += hv * pr[s]
    return (A_mix + A_lin + A_assoc) / nsite


@_njit(cache=True, parallel=True)
def _jacobian_logW_parallel(mu, y, phi, S, delta, A, e, lm, nsite, R, h_fd):
    """Finite-difference Jacobian with the (1 + F) columns evaluated in parallel.

    With F = 27 super-patch types at 3x3 this is where essentially all of the
    wall time goes, and the columns are completely independent.
    """
    Fd = y.shape[0]
    n_p = 1 + Fd
    Jac = np.empty((n_p, n_p))
    for j in prange(n_p):
        if j == 0:
            Rp, _, _, _, _ = _eval_residual_reduced_logW(
                mu + h_fd, y, phi, S, delta, A, e, lm, nsite)
        else:
            yp = y.copy()
            yp[j - 1] += h_fd
            Rp, _, _, _, _ = _eval_residual_reduced_logW(
                mu, yp, phi, S, delta, A, e, lm, nsite)
        for k in range(n_p):
            Jac[k, j] = (Rp[k] - R[k]) / h_fd
    return Jac


@_njit(cache=True)
def _solve_single_phi_logW_par(phi, mu0, W0, S, delta, A, e, lm, nsite,
                               max_newton=40, tol=1e-10):
    Fd = W0.shape[0]
    n_p = 1 + Fd
    h_fd = 1e-6

    y = np.empty(Fd)
    for s in range(Fd):
        y[s] = np.log(max(W0[s], 1e-6))

    mu = _find_initial_mu(phi, A, lm, e, S, delta, np.exp(y), nsite)

    it_done = 0
    for it in range(max_newton):
        it_done = it + 1
        R, rho, X, pr, W = _eval_residual_reduced_logW(mu, y, phi, S, delta, A, e, lm, nsite)
        rn = 0.0
        for k in range(n_p):
            rn += R[k] * R[k]
        rn = np.sqrt(rn)
        if rn < tol:
            break

        Jac = _jacobian_logW_parallel(mu, y, phi, S, delta, A, e, lm, nsite, R, h_fd)
        for k in range(n_p):
            Jac[k, k] += 1e-12

        dp = np.linalg.solve(Jac, -R)
        for k in range(n_p):
            if dp[k] > 10.0:
                dp[k] = 10.0
            elif dp[k] < -10.0:
                dp[k] = -10.0

        alpha = 1.0
        for _ in range(20):
            Rn, _, _, _, _ = _eval_residual_reduced_logW(
                mu + alpha * dp[0], y + alpha * dp[1:], phi, S, delta, A, e, lm, nsite)
            rn_ls = 0.0
            for k in range(n_p):
                rn_ls += Rn[k] * Rn[k]
            rn_ls = np.sqrt(rn_ls)
            if rn_ls < rn:
                break
            alpha *= 0.5
        mu += alpha * dp[0]
        y += alpha * dp[1:]

    R, rho, X, pr, W = _eval_residual_reduced_logW(mu, y, phi, S, delta, A, e, lm, nsite)
    rn_final = 0.0
    for k in range(n_p):
        rn_final += R[k] * R[k]
    rn_final = np.sqrt(rn_final)
    f = _free_energy_from_state(rho, X, pr, A, e, lm, nsite)
    return f, mu, W, rn_final, it_done


@_njit(cache=True)
def _solve_single_phi_logW(phi, mu0, W0, S, delta, A, e, lm, nsite,
                           max_newton=40, tol=1e-10):
    Fd = W0.shape[0]
    n_p = 1 + Fd
    h_fd = 1e-6

    y = np.empty(Fd)
    for s in range(Fd):
        y[s] = np.log(max(W0[s], 1e-6))

    W_seed = np.exp(y)
    mu = _find_initial_mu(phi, A, lm, e, S, delta, W_seed, nsite)

    it_done = 0
    for it in range(max_newton):
        it_done = it + 1
        R, rho, X, pr, W = _eval_residual_reduced_logW(mu, y, phi, S, delta, A, e, lm, nsite)
        rn = 0.0
        for k in range(n_p):
            rn += R[k] * R[k]
        rn = np.sqrt(rn)
        if rn < tol:
            break

        Jac = np.empty((n_p, n_p))
        Rp, _, _, _, _ = _eval_residual_reduced_logW(mu + h_fd, y, phi, S, delta, A, e, lm, nsite)
        for k in range(n_p):
            Jac[k, 0] = (Rp[k] - R[k]) / h_fd
        for j in range(Fd):
            yp = y.copy()
            yp[j] += h_fd
            Rp2, _, _, _, _ = _eval_residual_reduced_logW(mu, yp, phi, S, delta, A, e, lm, nsite)
            for k in range(n_p):
                Jac[k, 1 + j] = (Rp2[k] - R[k]) / h_fd
        for k in range(n_p):
            Jac[k, k] += 1e-12

        dp = np.linalg.solve(Jac, -R)
        for k in range(n_p):
            if dp[k] > 10.0:
                dp[k] = 10.0
            elif dp[k] < -10.0:
                dp[k] = -10.0

        alpha = 1.0
        for _ in range(20):
            mn = mu + alpha * dp[0]
            yn = y + alpha * dp[1:]
            Rn, _, _, _, _ = _eval_residual_reduced_logW(mn, yn, phi, S, delta, A, e, lm, nsite)
            rn_ls = 0.0
            for k in range(n_p):
                rn_ls += Rn[k] * Rn[k]
            rn_ls = np.sqrt(rn_ls)
            if rn_ls < rn:
                break
            alpha *= 0.5
        mu += alpha * dp[0]
        y += alpha * dp[1:]

    R, rho, X, pr, W = _eval_residual_reduced_logW(mu, y, phi, S, delta, A, e, lm, nsite)
    rn_final = 0.0
    for k in range(n_p):
        rn_final += R[k] * R[k]
    rn_final = np.sqrt(rn_final)
    f = _free_energy_from_state(rho, X, pr, A, e, lm, nsite)
    return f, mu, W, rn_final, it_done


@_njit(cache=True)
def _solve_single_phi_directW(phi, mu0, W0, S, delta, A, e, lm, nsite,
                              max_newton=80, tol=1e-10):
    Fd = W0.shape[0]
    n_p = 1 + Fd
    h_fd = 1e-7
    mu = mu0
    W = W0.copy()

    R0, _, _, _ = _eval_residual_reduced(mu, W, phi, S, delta, A, e, lm, nsite)
    rn0 = 0.0
    for k in range(n_p):
        rn0 += R0[k] * R0[k]
    rn0 = np.sqrt(rn0)
    if rn0 > 0.1:
        mu = _find_initial_mu(phi, A, lm, e, S, delta, W, nsite)

    it_done = 0
    for it in range(max_newton):
        it_done = it + 1
        R, rho, X, pr = _eval_residual_reduced(mu, W, phi, S, delta, A, e, lm, nsite)
        rn = 0.0
        for k in range(n_p):
            rn += R[k] * R[k]
        rn = np.sqrt(rn)
        if rn < tol:
            break

        Jac = np.empty((n_p, n_p))
        Rp, _, _, _ = _eval_residual_reduced(mu + h_fd, W, phi, S, delta, A, e, lm, nsite)
        for k in range(n_p):
            Jac[k, 0] = (Rp[k] - R[k]) / h_fd
        for j in range(Fd):
            Wp = W.copy()
            Wp[j] += h_fd
            Rp2, _, _, _ = _eval_residual_reduced(mu, Wp, phi, S, delta, A, e, lm, nsite)
            for k in range(n_p):
                Jac[k, 1 + j] = (Rp2[k] - R[k]) / h_fd
        for k in range(n_p):
            Jac[k, k] += 1e-12

        dp = np.linalg.solve(Jac, -R)
        for k in range(n_p):
            if dp[k] > 10.0:
                dp[k] = 10.0
            elif dp[k] < -10.0:
                dp[k] = -10.0

        alpha = 1.0
        for _ in range(20):
            mn = mu + alpha * dp[0]
            Wn = W + alpha * dp[1:]
            Rn, _, _, _ = _eval_residual_reduced(mn, Wn, phi, S, delta, A, e, lm, nsite)
            rn_ls = 0.0
            for k in range(n_p):
                rn_ls += Rn[k] * Rn[k]
            rn_ls = np.sqrt(rn_ls)
            if rn_ls < rn:
                break
            alpha *= 0.5
        mu += alpha * dp[0]
        W += alpha * dp[1:]

    R, rho, X, pr = _eval_residual_reduced(mu, W, phi, S, delta, A, e, lm, nsite)
    rn_final = 0.0
    for k in range(n_p):
        rn_final += R[k] * R[k]
    rn_final = np.sqrt(rn_final)
    f = _free_energy_from_state(rho, X, pr, A, e, lm, nsite)
    return f, mu, W, rn_final, it_done


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------

def solve_free_energy_curve(
    phi_grid, A_row, e_linear, log_mult, S, delta, nsite,
    *, mu_init=0.0, W_init=None, max_newton=40, tol=1e-9,
    accept_residual=1e-6, logW=True, parallel=False,
):
    phi_grid = np.asarray(phi_grid, dtype=np.float64)
    A_row = np.ascontiguousarray(np.asarray(A_row, dtype=np.float64))
    e_linear = np.ascontiguousarray(np.asarray(e_linear, dtype=np.float64))
    log_mult = np.ascontiguousarray(np.asarray(log_mult, dtype=np.float64))
    S = np.ascontiguousarray(np.asarray(S, dtype=np.float64))
    delta = np.ascontiguousarray(np.asarray(delta, dtype=np.float64))
    nsite = float(nsite)

    Fd = S.shape[1]
    phi_lo, phi_hi = float(A_row.min()), float(A_row.max())

    fvals = np.full(len(phi_grid), np.nan)
    mus = np.full(len(phi_grid), np.nan)
    resid = np.full(len(phi_grid), np.nan)
    iters = np.zeros(len(phi_grid), dtype=np.int64)

    mu = float(mu_init)
    W = (np.full(Fd, 1e-3) if W_init is None
         else np.asarray(W_init, dtype=np.float64).copy())

    if not logW:
        solver = _solve_single_phi_directW
    elif parallel and _HAS_NUMBA:
        solver = _solve_single_phi_logW_par
    else:
        solver = _solve_single_phi_logW
    for k, phi in enumerate(phi_grid):
        phi = float(phi)
        if phi < phi_lo - 1e-12 or phi > phi_hi + 1e-12:
            continue
        f, mu, W, res, nit = solver(phi, mu, W, S, delta, A_row, e_linear, log_mult,
                                    nsite, max_newton, tol)
        mus[k] = mu
        resid[k] = res
        iters[k] = nit
        if np.isfinite(f) and res <= accept_residual:
            fvals[k] = f

    return {"phi": phi_grid, "f": fvals, "mus": mus, "residual": resid,
            "iters": iters, "mu_W": (float(mu), W.copy())}
