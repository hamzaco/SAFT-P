import numpy as np
import torch
from typing import Optional, Tuple, Dict, Any, List
import torch.nn.functional as F

from plaquette_composition import (
    COMPOSITION_KEYS,
    resolve_component_map,
    composition_codes_bulk,
    describe_components,
)

try:
    from numba import njit as _njit
    _HAS_NUMBA = True
except ImportError:
    _HAS_NUMBA = False
    def _njit(*args, **kwargs):
        def _wrap(fn):
            return fn
        if args and callable(args[0]):
            return args[0]
        return _wrap

def _rot90_species_map(n_species: int, rot90_species: Optional[np.ndarray]) -> np.ndarray:
    """
    rot90_species: permutation of [0..S-1] giving the species after a 90° CW rotation.
    """
    if rot90_species is None:
        return np.arange(n_species, dtype=np.int64)

    rot90_species = np.asarray(rot90_species, dtype=np.int64)
    if rot90_species.shape != (n_species,):
        raise ValueError(f"rot90_species must have shape ({n_species},), got {rot90_species.shape}")

    if rot90_species.min() < 0 or rot90_species.max() >= n_species:
        raise ValueError("rot90_species entries must be in [0, n_species).")

    if np.unique(rot90_species).size != n_species:
        raise ValueError("rot90_species must be a permutation (each species appears exactly once).")

    return rot90_species


def _rotate_plaq_cfg(
    cfg: Tuple[int, int, int, int], rot90_species: np.ndarray, k: int
) -> Tuple[int, int, int, int]:
    """
    Rotate a plaquette configuration by k quarter-turns (90° clockwise each),
    applying the per-species 90° rotation map to each corner species.

    Corner order convention used throughout:
      (0,1,2,3) = (UL, UR, BL, BR)

    Under a 90° CW global rotation, corners map:
      new UL <- old BL
      new UR <- old UL
      new BL <- old BR
      new BR <- old UR

    So:
      (sUL, sUR, sBL, sBR) -> (rot(sBL), rot(sUL), rot(sBR), rot(sUR))
    """
    sUL, sUR, sBL, sBR = cfg
    for _ in range(int(k) % 4):
        sUL, sUR, sBL, sBR = (
            int(rot90_species[sBL]),
            int(rot90_species[sUL]),
            int(rot90_species[sBR]),
            int(rot90_species[sUR]),
        )
    return (sUL, sUR, sBL, sBR)


def _canonical_key_rotation_only(
    cfg: Tuple[int, int, int, int], rot90_species: np.ndarray
) -> Tuple[Tuple[int, int, int, int], int]:
    """
    Canonicalize cfg up to rotation only (NOT reflection), returning:
      - canonical representative tuple
      - multiplicity = number of unique rotations collapsed (1,2,4)
    """
    rots = [_rotate_plaq_cfg(cfg, rot90_species, k) for k in range(4)]
    uniq = sorted(set(rots))
    canon = uniq[0]
    mult = len(uniq)
    return canon, mult


def _get_edge_structure(
    cfg: Tuple[int, int, int, int],
    patches: np.ndarray,
    M: int,
) -> Tuple[int, int, int, int]:
    """
    Compute edge structure (top, right, bottom, left) for a plaquette configuration.
    
    Args:
      cfg: (sUL, sUR, sBL, sBR) species configuration
      patches: (S, 4) array of patch assignments per species
      M: number of patch types
    
    Returns:
      (top_edge, right_edge, bottom_edge, left_edge) as edge IDs
    """
    sUL, sUR, sBL, sBR = cfg
    N, E, Sdir, W = 0, 1, 2, 3
    
    pUL = patches[sUL]
    pUR = patches[sUR]
    pBL = patches[sBL]
    pBR = patches[sBR]
    
    top_edge = pUL[N] * M + pUR[N]
    right_edge = pUR[E] * M + pBR[E]
    bottom_edge = pBR[Sdir] * M + pBL[Sdir]
    left_edge = pBL[W] * M + pUL[W]
    
    return (top_edge, right_edge, bottom_edge, left_edge)


def _canonical_key_by_edges(
    cfg: Tuple[int, int, int, int],
    patches: np.ndarray,
    M: int,
) -> Tuple[Tuple[int, int, int, int], int]:
    """
    Canonicalize cfg by edge structure (not by constituent species).
    Two plaquettes are equivalent if they have the same edge structure up to rotation.
    
    Returns:
      - canonical edge structure tuple (top, right, bottom, left)
      - multiplicity = number of unique rotations collapsed (1,2,4)
    """
    edges = _get_edge_structure(cfg, patches, M)
    
    # Generate all 4 rotations of the edge structure
    top, right, bottom, left = edges
    rots = [
        (top, right, bottom, left),           # 0° rotation
        (left, top, right, bottom),          # 90° CW: (top,right,bottom,left) -> (left,top,right,bottom)
        (bottom, left, top, right),           # 180°: (top,right,bottom,left) -> (bottom,left,top,right)
        (right, bottom, left, top),           # 270°: (top,right,bottom,left) -> (right,bottom,left,top)
    ]
    
    uniq = sorted(set(rots))
    canon = uniq[0]
    mult = len(uniq)
    return canon, mult



def residual_free_energy_by_species(
    rho: torch.Tensor,
    beta: float,
    z_assoc: float,
    patch_to_species: torch.Tensor,
    patch_to_small: torch.Tensor,
    eps_small: torch.Tensor,
    m_patch: torch.Tensor,
    e_linear: torch.Tensor,
    delta_small: torch.Tensor,
    mult: torch.Tensor,
    *,
    X0: Optional[torch.Tensor] = None,
    return_X: bool = False,
    max_iter: int = 1000,
    tol: float = 1e-5,
    damp: float = 0.7,
) -> torch.Tensor:
    """
    Dimensionless free energy (beta*F) per plaquette:

      A_mix   = sum_i rho_i * log rho_i
      A_lin   = dot(e_linear, rho)
      A_assoc = sum_p m_p * phi_p * (log X_p - 0.5 X_p + 0.5)

    Note: Multiplicity is now included in Boltzmann weights during plaquette construction,
    so it is not subtracted from the mixing entropy here.

    z_assoc multiplies the mass-action term: denom = 1 + z_assoc * u[patch_to_small].
    """
    _ = beta  # beta can be absorbed into delta_small upstream; keep for API stability.
    _ = mult  # Multiplicity is now included in Boltzmann weights, not used here

    eps = torch.tensor(1e-12, dtype=rho.dtype, device=rho.device)

    # Mixing (multiplicity already accounted for in Boltzmann weights)
    A_mix = torch.sum(rho * torch.log(rho + eps))

    # Linear
    A_lin = torch.dot(e_linear, rho)

    # Association fixed-point
    M = int(eps_small.shape[0])
    phi_p = rho[patch_to_species]

    if X0 is None:
        X = torch.ones_like(phi_p)
    else:
        X = X0.to(dtype=rho.dtype, device=rho.device).clone()
        if X.shape != phi_p.shape:
            raise ValueError(f"X0 must have shape {tuple(phi_p.shape)}, got {tuple(X.shape)}")

    for _ in range(max_iter):
        W = torch.zeros(M, dtype=rho.dtype, device=rho.device)
        W.index_add_(0, patch_to_small, m_patch * phi_p * X)

        u = delta_small.matmul(W)  # (M,)

        denom = (1.0 + u[patch_to_small]).clamp_min(1e-12)
        X_new = (1.0 / denom).clamp(1e-12, 1.0)

        if torch.max(torch.abs(X_new - X)) < tol:
            X = X_new
            break

        X = damp * X_new + (1.0 - damp) * X

    A_assoc = torch.sum(m_patch * phi_p * (torch.log(X + eps) - 0.5 * X + 0.5))

    f = (A_lin + A_assoc + A_mix) / 4.0
    if return_X:
        return f, X
    return f


def build_A_rows_from_plaq(plaq_to_species):
    """Build A rows from plaquette-to-species mapping."""
    return plaq_to_species.T.copy()
# --------------------------------------------------------------------------
# 3) FIXED: free_energy_curve_chirality
#    - warm-start uses z directly (no log(rho) branch-jump)
# --------------------------------------------------------------------------
def free_energy_curve_chirality(
    phi_grid, A_rows, targets_template, variable_idx, f_tilde_torch, *,
    device="cpu", dtype=torch.float64, steps=400, lr=0.5, tol_res=1e-4,
):
    """
    Compute f(phi) curve; warm-start in z-space across phi using the previous z.
    """
    phi_grid = np.asarray(phi_grid, float)
    fvals, rhos = [], []
    z0 = None

    for phi in phi_grid:
        targets = targets_template.copy()
        targets[variable_idx] = phi

        out = monomer_free_energy_general(
            A_rows, targets, f_tilde_torch,
            steps=steps, lr=lr, tol_res=tol_res, device=device, dtype=dtype, z0=z0,
        )
        fvals.append(out["f_phi"])
        rhos.append(out["rho"])
        z0 = out["z"]  # <-- FIXED warm-start

    fvals_b = np.asarray(fvals, float)
    
    fvals, rhos = [], []
    for phi in phi_grid[::-1]:
        targets = targets_template.copy()
        targets[variable_idx] = phi

        out = monomer_free_energy_general(
            A_rows, targets, f_tilde_torch,
            steps=steps, lr=lr, tol_res=tol_res, device=device, dtype=dtype, z0=z0,
        )
        fvals.append(out["f_phi"])
        rhos.append(out["rho"])
        z0 = out["z"]  # <-- FIXED warm-start

    fvals_f = np.asarray(fvals, float)[::-1]
    fvals= np.minimum(fvals_f,fvals_b)
    # Smooth derivatives as before
    f_smooth_full, fp_full, fpp_full = loess_derivs(
        torch.tensor(phi_grid), torch.tensor(fvals), bandwidth=0.2
    )

    phi_inner = phi_grid[2:-2]
    f_smooth = f_smooth_full[2:-2]
    fp = fp_full[2:-2]
    fpp = fpp_full[2:-2]

    # Spinodal from sign changes of -fpp (your convention)
    spinodal_phis = []
    fpp_np = (-fpp).numpy()
    for i in range(len(phi_inner) - 1):
        y1, y2 = fpp_np[i], fpp_np[i + 1]
        if y1 * y2 < 0:
            t = -y1 / (y2 - y1)
            spinodal_phis.append(float(phi_inner[i] + t * (phi_inner[i + 1] - phi_inner[i])))

    return dict(
        phi=phi_grid,
        f=fvals,
        phi_inner=phi_inner,
        f_smooth=f_smooth,
        fp=fp,
        fpp=fpp,
        spinodal_phis=spinodal_phis,
        rhos=rhos,
    )



def find_feasible_logits(
    C: torch.Tensor,
    d: torch.Tensor,
    *,
    max_iter: int = 200,
    tol: float = 1e-10,
    ridge: float = 1e-12,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """
    Primal feasibility solve:
        minimize 0.5 * || C softmax(y) - d ||^2  over logits y (P,)
    Always well-defined; if infeasible, returns best residual.

    Returns: (y, rho, res)
    """
    m, P = C.shape
    y = torch.zeros(P, dtype=C.dtype, device=C.device, requires_grad=True)
    opt = torch.optim.LBFGS([y], max_iter=max_iter, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        rho = F.softmax(y, dim=0)
        g = C @ rho - d
        loss = 0.5 * (g @ g)
        # mild ridge to prevent y blowup
        loss = loss + 0.5 * ridge * (y @ y)
        loss.backward()
        return loss

    opt.step(closure)

    with torch.no_grad():
        rho = F.softmax(y, dim=0)
        g = C @ rho - d
        res = float(torch.linalg.norm(g).item())
    return y.detach(), rho.detach(), res


def constrained_softmax(
    z: torch.Tensor,
    C: torch.Tensor,
    d: torch.Tensor,
    *,
    tol: float = 1e-10,
    max_newton: int = 150,
    ridge: float = 1e-10,
    max_ls: int = 20,
    use_lbfgs_fallback: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """
    Solve rho = softmax(z + C^T eta) s.t. C @ rho = d.

    Returns: (rho, eta, res_norm).
    If infeasible or ill-conditioned, res_norm will stay large.
    """
    m, P = C.shape
    dtype = z.dtype
    device = z.device

    eta = torch.zeros(m, dtype=dtype, device=device)

    def eval_res(eta_vec: torch.Tensor):
        logits = z + C.t().matmul(eta_vec)
        # stabilize: subtract max(logits) (softmax already does this, but keep explicit)
        logits = logits - torch.max(logits)
        rho = F.softmax(logits, dim=0)
        g = C.matmul(rho) - d
        res = torch.linalg.norm(g)
        return rho, g, float(res.item())

    rho, g, res = eval_res(eta)
    best_eta = eta.clone()
    best_res = res

    for _ in range(max_newton):
        if res <= tol:
            return rho, eta, res

        Cr = C.matmul(rho)  # (m,)
        W = C * rho.unsqueeze(0)              # (m,P)
        J = W.matmul(C.t()) - torch.outer(Cr, Cr)
        J = J + ridge * torch.eye(m, dtype=dtype, device=device)

        try:
            delta = torch.linalg.solve(J, -g)
        except RuntimeError:
            J = J + (10.0 * ridge) * torch.eye(m, dtype=dtype, device=device)
            delta = torch.linalg.solve(J, -g)

        # backtracking on ||g||^2
        step = 1.0
        base = res * res
        accepted = False
        for _ls in range(max_ls):
            eta_new = eta + step * delta
            rho_new, g_new, res_new = eval_res(eta_new)
            if res_new < best_res:
                best_res = res_new
                best_eta = eta_new.clone()
            if res_new * res_new <= base * (1.0 - 1e-4 * step):
                eta, rho, g, res = eta_new, rho_new, g_new, res_new
                accepted = True
                break
            step *= 0.5

        if not accepted:
            eta = best_eta
            rho, g, res = eval_res(eta)
            break

    # Fallback: LBFGS on eta to minimize 0.5 ||C softmax(z + C^T eta) - d||^2
    if use_lbfgs_fallback and res > tol:
        eta_var = best_eta.clone().detach().requires_grad_(True)
        opt = torch.optim.LBFGS([eta_var], max_iter=80, line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad()
            logits = z + C.t().matmul(eta_var)
            logits = logits - torch.max(logits)
            rho_ = F.softmax(logits, dim=0)
            g_ = C.matmul(rho_) - d
            loss = 0.5 * (g_ @ g_)
            loss.backward()
            return loss

        try:
            opt.step(closure)
            eta = eta_var.detach()
            rho, g, res = eval_res(eta)
        except Exception:
            eta = best_eta
            rho, g, res = eval_res(eta)

    return rho, eta, res


def _stack_constraints_general(rows, targets, G=None):
    rows = np.atleast_2d(rows).astype(np.float64)
    targets = np.atleast_1d(targets).astype(np.float64)
    if rows.shape[0] != targets.shape[0]:
        raise ValueError(f"rows has {rows.shape[0]} constraints but targets has {targets.shape[0]}")
    C = rows
    d = targets
    if G is not None:
        C = np.vstack([C, G.astype(np.float64)])
        d = np.concatenate([d, np.zeros(G.shape[0], dtype=np.float64)])
    return C, d


def monomer_free_energy_general(
    A_rows, targets, f_tilde_torch, *,
    G=None, steps=8000, lr=0.5, tol_res=1e-6,
    device="cpu", dtype=torch.float64, z0=None,
    reinit_if_res_gt: float = 1e-3,        # if projection residual exceeds this, reinitialize z
    max_reinits: int = 5,                  # how many reinitializations allowed
    feasibility_tol: float = 1e-2,         # if primal feasibility can't reach this, treat constraints as infeasible
):
    """
    Key fix: do NOT proceed when constraint projection fails.
    Reinitialize logits using primal feasibility solve when residual is large.
    """
    C_np, d_np = _stack_constraints_general(A_rows, targets, G)
    C = torch.tensor(C_np, dtype=dtype, device=device)
    d = torch.tensor(d_np, dtype=dtype, device=device)
    P = C.shape[1]

    # Initialize z
    if z0 is None:
        y_feas, rho_feas, res_feas = find_feasible_logits(C, d, tol=tol_res)
        if res_feas > feasibility_tol:
            raise RuntimeError(f"Infeasible constraints (best residual {res_feas:.3e}). Integer plaquettes cannot realize this target.")
        z_init = y_feas
    else:
        z_init = z0.clone().detach()

    z = z_init.requires_grad_(True)
    opt = torch.optim.Adagrad([z], lr=lr)

    prev_f = None
    reinits = 0

    for _ in range(steps):
        rho, _, resn = constrained_softmax(z, C, d, tol=tol_res)

        # If projection failed, reinitialize z from primal feasibility
        if resn > reinit_if_res_gt:
            reinits += 1
            y_feas, rho_feas, res_feas = find_feasible_logits(C, d, tol=tol_res)
            if res_feas > feasibility_tol or reinits > max_reinits:
                raise RuntimeError(
                    f"Constraint projection failing / infeasible. "
                    f"proj_res={resn:.3e}, feas_res={res_feas:.3e}, reinits={reinits}"
                )
            with torch.no_grad():
                z.copy_(y_feas)
            continue

        f = f_tilde_torch(rho)
        opt.zero_grad()
        f.backward()
        opt.step()

        with torch.no_grad():
            rho2, _, res2 = constrained_softmax(z, C, d, tol=tol_res)
            if res2 > reinit_if_res_gt:
                # force a reinit next loop
                prev_f = None
                continue
            f2 = float(f_tilde_torch(rho2).item())

        if prev_f is not None and abs(f2 - prev_f) <= tol_res:
            break
        prev_f = f2

    with torch.no_grad():
        rho_star, _, res = constrained_softmax(z, C, d, tol=tol_res)
        f_phi = float(f_tilde_torch(rho_star).item())
        # print both C@rho and residual so you stop misreading "all zeros"
        Cr = C @ rho_star
        print(Cr)

    return {"f_phi": f_phi, "rho": rho_star.detach().cpu().numpy(), "residual": float(res), "z": z.detach()}
#-------------------------------------------

def loess_derivs(x, y, *, bandwidth=0.15, order=2):
    x = torch.as_tensor(x, dtype=torch.float64)
    y = torch.as_tensor(y, dtype=torch.float64)
    n = x.shape[0]
    f0 = torch.zeros(n, dtype=torch.float64)
    f1 = torch.zeros(n, dtype=torch.float64)
    f2 = torch.zeros(n, dtype=torch.float64)
    h = bandwidth * (x.max() - x.min())
    for i in range(n):
        d = (x - x[i]).abs()
        w = torch.clamp(1 - (d / h) ** 3, min=0) ** 3
        dx = x - x[i]
        cols = [torch.ones_like(dx)]
        if order >= 1:
            cols.append(dx)
        if order >= 2:
            cols.append(dx ** 2)
        X = torch.stack(cols, dim=1)
        XTW = (X.T * w)
        beta = torch.linalg.solve(XTW @ X, XTW @ y)
        f0[i] = beta[0]
        f1[i] = beta[1] if order >= 1 else torch.nan
        f2[i] = (2 * beta[2]) if order >= 2 else torch.nan
    return f0, f1, f2


def _stack_constraints(A, phi, G):
    A = np.atleast_2d(A).astype(np.float64)
    rows = [A]
    rhs = [np.array([phi], dtype=np.float64)]
    if G is not None:
        rows.append(G.astype(np.float64))
        rhs.append(np.zeros(G.shape[0], dtype=np.float64))
    return np.vstack(rows), np.concatenate(rhs)


class StepTracker:
    def __init__(self):
        self.data = []
    def log(self, **kwargs):
        self.data.append(kwargs)
    def as_list(self):
        return self.data


def build_plaquettes_by_species(
    patches: np.ndarray,
    J: np.ndarray,
    mu: Optional[np.ndarray] = None,
    rot90_species: Optional[np.ndarray] = None,
    *,
    compress_boundary: bool = True,
    use_4_patch: bool = True,
    undirected_edges: bool = False,
    canonicalize_by_edges: bool = False,
    canonicalize_by_boundary_edges: bool = False,   # NEW
    return_class_members: bool = False,
) -> Any:

    if canonicalize_by_edges and canonicalize_by_boundary_edges:
        raise ValueError("Pick one: canonicalize_by_edges OR canonicalize_by_boundary_edges, not both.")

    patches = np.asarray(patches, dtype=np.int64)
    J = np.asarray(J, dtype=np.float64)

    Ssp = int(patches.shape[0])
    M = int(J.shape[0])

    if J.shape != (M, M):
        raise ValueError(f"J must be square, got {J.shape}")

    if patches.max() >= M or patches.min() < 0:
        raise ValueError("patch ids in patches must be within [0, J.shape[0])")

    mu_vec = np.zeros(Ssp, dtype=np.float64) if mu is None else np.asarray(mu, dtype=np.float64)
    if mu_vec.shape != (Ssp,):
        raise ValueError(f"mu must have shape ({Ssp},), got {mu_vec.shape}")

    rot_map = _rot90_species_map(Ssp, rot90_species)

    N, E, Sdir, W = 0, 1, 2, 3

    def cfg_energy(cfg: Tuple[int, int, int, int]) -> float:
        sUL, sUR, sBL, sBR = cfg
        pUL = patches[sUL]; pUR = patches[sUR]; pBL = patches[sBL]; pBR = patches[sBR]
        e = 0.0
        e += J[pUL[E],    pUR[W]]
        e += J[pBL[E],    pBR[W]]
        e += J[pUL[Sdir], pBL[N]]
        e += J[pUR[Sdir], pBR[N]]
        return float(e)

    def cfg_mu(cfg: Tuple[int, int, int, int]) -> float:
        sUL, sUR, sBL, sBR = cfg
        return float(mu_vec[sUL] + mu_vec[sUR] + mu_vec[sBL] + mu_vec[sBR])

    def cfg_species_counts(cfg: Tuple[int, int, int, int]) -> np.ndarray:
        sUL, sUR, sBL, sBR = cfg
        c = np.zeros(Ssp, dtype=np.float64)
        c[sUL] += 1.0; c[sUR] += 1.0; c[sBL] += 1.0; c[sBR] += 1.0
        return c

    def _edge_id(a: int, b: int) -> int:
        a = int(a); b = int(b)
        if undirected_edges and a > b:
            a, b = b, a
        return a * M + b

    def cfg_boundary_counts(cfg: Tuple[int, int, int, int]) -> Dict[int, float]:
        sUL, sUR, sBL, sBR = cfg
        pUL = patches[sUL]; pUR = patches[sUR]; pBL = patches[sBL]; pBR = patches[sBR]

        if use_4_patch:
            top_edge    = _edge_id(pUL[N],    pUR[N])
            right_edge  = _edge_id(pUR[E],    pBR[E])
            bottom_edge = _edge_id(pBR[Sdir], pBL[Sdir])
            left_edge   = _edge_id(pBL[W],    pUL[W])
            out: Dict[int, float] = {}
            out[top_edge]    = out.get(top_edge, 0.0) + 1.0
            out[right_edge]  = out.get(right_edge, 0.0) + 1.0
            out[bottom_edge] = out.get(bottom_edge, 0.0) + 1.0
            out[left_edge]   = out.get(left_edge, 0.0) + 1.0
            return out

        # directed-boundary branch
        b = np.array([
            0*M + int(pUL[N]),    0*M + int(pUR[N]),
            1*M + int(pUR[E]),    1*M + int(pBR[E]),
            2*M + int(pBR[Sdir]), 2*M + int(pBL[Sdir]),
            3*M + int(pBL[W]),    3*M + int(pUL[W]),
        ], dtype=np.int64)

        if compress_boundary:
            uniq, cnt = np.unique(b, return_counts=True)
            return {int(u): float(c) for u, c in zip(uniq, cnt)}

        out: Dict[int, float] = {}
        for t in b.tolist():
            out[int(t)] = out.get(int(t), 0.0) + 1.0
        return out

    def unique_rots(cfg: Tuple[int, int, int, int]):
        rots = [_rotate_plaq_cfg(cfg, rot_map, k) for k in range(4)]
        return list(sorted(set(rots)))

    def new_acc():
        return {
            "E0": np.inf,
            "w": 0.0,
            "species_sum": np.zeros(Ssp, dtype=np.float64),
            "boundary_sum": {},
            "best_cfg": None,
        }

    def acc_add(acc: Dict[str, Any], cfg: Tuple[int, int, int, int]):
        E = cfg_energy(cfg)

        # include mu only when you merge across compositions (either canonicalization mode)
        if canonicalize_by_edges or canonicalize_by_boundary_edges:
            Eeff = float(E - cfg_mu(cfg))
        else:
            Eeff = float(E)

        # maintain E0 shift so weights stay O(1)
        if Eeff < acc["E0"]:
            if np.isfinite(acc["E0"]):
                fac = float(np.exp(-(acc["E0"] - Eeff)))
                acc["w"] *= fac
                acc["species_sum"] *= fac
                for k in list(acc["boundary_sum"].keys()):
                    acc["boundary_sum"][k] *= fac
            acc["E0"] = Eeff
            acc["best_cfg"] = cfg

        contrib = float(np.exp(-(Eeff - acc["E0"])))
        acc["w"] += contrib
        acc["species_sum"] += contrib * cfg_species_counts(cfg)

        bcounts = cfg_boundary_counts(cfg)
        bsum = acc["boundary_sum"]
        for t, cnt in bcounts.items():
            bsum[int(t)] = bsum.get(int(t), 0.0) + contrib * float(cnt)

    # ---- enumerate unique rotation orbits ----
    seen_orbits = set()
    orbit_reps = []
    for s0 in range(Ssp):
        for s1 in range(Ssp):
            for s2 in range(Ssp):
                for s3 in range(Ssp):
                    cfg = (s0, s1, s2, s3)
                    rots = unique_rots(cfg)
                    orbit = tuple(rots)
                    if orbit in seen_orbits:
                        continue
                    seen_orbits.add(orbit)
                    orbit_reps.append(rots[0])

    def _canonical_key_by_internal_edges(cfg: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
        rots = unique_rots(cfg)
        keys = []
        for r in rots:
            sUL, sUR, sBL, sBR = r
            pUL = patches[sUL]; pUR = patches[sUR]; pBL = patches[sBL]; pBR = patches[sBR]
            top    = _edge_id(pUL[E],    pUR[W])    # internal
            right  = _edge_id(pUR[Sdir], pBR[N])    # internal
            bottom = _edge_id(pBL[E],    pBR[W])    # internal
            left   = _edge_id(pUL[Sdir], pBL[N])    # internal
            keys.append((top, right, bottom, left))
        return min(keys)

    def _canonical_key_by_boundary_edges(cfg: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
        if not use_4_patch:
            raise ValueError("canonicalize_by_boundary_edges requires use_4_patch=True (perimeter edge ids).")
        rots = unique_rots(cfg)
        keys = []
        for r in rots:
            sUL, sUR, sBL, sBR = r
            pUL = patches[sUL]; pUR = patches[sUR]; pBL = patches[sBL]; pBR = patches[sBR]
            top    = _edge_id(pUL[N],    pUR[N])        # boundary
            right  = _edge_id(pUR[E],    pBR[E])        # boundary
            bottom = _edge_id(pBR[Sdir], pBL[Sdir])     # boundary
            left   = _edge_id(pBL[W],    pUL[W])        # boundary
            keys.append((top, right, bottom, left))
        return min(keys)

    groups: Dict[Any, Dict[str, Any]] = {}
    for rep in orbit_reps:
        rots = unique_rots(rep)

        if canonicalize_by_boundary_edges:
            key = _canonical_key_by_boundary_edges(rots[0])
        elif canonicalize_by_edges:
            key = _canonical_key_by_internal_edges(rots[0])
        else:
            key = tuple(rots)

        acc = groups.get(key)
        if acc is None:
            acc = new_acc()
            groups[key] = acc
        for r in rots:
            acc_add(acc, r)

    # ---- finalize macrostates ----
    reps = []
    intra_bonds = []
    mult_arr = []
    plaq_to_species_rows = []
    boundary_avg_list = []

    for _, acc in groups.items():
        w = float(acc["w"])
        if (not np.isfinite(acc["E0"])) or (w <= 0.0):
            continue

        reps.append(acc["best_cfg"])
        intra_bonds.append(float(acc["E0"]))

        species_avg = acc["species_sum"] / w
        plaq_to_species_rows.append(species_avg)

        bavg = {int(t): float(s / w) for t, s in acc["boundary_sum"].items()}
        boundary_avg_list.append(bavg)

        # CONSISTENT WITH BOLTZMANN AVERAGING:
        # degeneracy is already in w, so multiplicity must be 1 for both canonicalization modes.
        if canonicalize_by_edges or canonicalize_by_boundary_edges:
            mult_arr.append(1.0)
        else:
            mult_arr.append(float(len(unique_rots(acc["best_cfg"]))))

    plaq_configs    = np.asarray(reps, dtype=np.int64)
    intra_bonds     = np.asarray(intra_bonds, dtype=np.float64)
    plaq_to_species = np.asarray(plaq_to_species_rows, dtype=np.float64)
    mult_arr        = np.asarray(mult_arr, dtype=np.float64)

    P = int(plaq_configs.shape[0])

    plaq_class_members: List[Tuple[Tuple[int, int, int, int], ...]] = []
    if return_class_members:
        # Plotting classes are defined independently of thermodynamic coarse-graining:
        # for use_4_patch, group rotation-canonical plaquettes by the same boundary-edge
        # signature (up to global rotation). This exposes all class members in motif plots
        # without changing the Hessian/optimization basis.
        def _plot_class_key(cfg: Tuple[int, int, int, int]) -> Any:
            if not use_4_patch:
                return tuple(unique_rots(cfg))
            return _canonical_key_by_boundary_edges(cfg)

        key_of_cfg: List[Any] = []
        class_map: Dict[Any, List[Tuple[int, int, int, int]]] = {}
        for cfg_arr in plaq_configs:
            cfg_t = tuple(int(x) for x in cfg_arr.tolist())
            k = _plot_class_key(cfg_t)
            key_of_cfg.append(k)
            class_map.setdefault(k, []).append(cfg_t)

        class_map_unique: Dict[Any, Tuple[Tuple[int, int, int, int], ...]] = {}
        for k, members in class_map.items():
            uniq_sorted = tuple(sorted(set(members)))
            class_map_unique[k] = uniq_sorted

        plaq_class_members = [class_map_unique[k] for k in key_of_cfg]

    # ---- build patch_to_species / patch_to_small / m_patch and eps_small ----
    if use_4_patch:
        all_edge_ids = sorted({int(t) for bavg in boundary_avg_list for t in bavg.keys()})
        unique_edge_ids = np.asarray(all_edge_ids, dtype=np.int64)
        edge_id_to_idx = {int(eid): idx for idx, eid in enumerate(unique_edge_ids.tolist())}

        pts, pss, mps = [], [], []
        for i in range(P):
            bavg = boundary_avg_list[i]
            for eid, cnt in bavg.items():
                pts.append(i)
                pss.append(edge_id_to_idx[int(eid)])
                mps.append(float(cnt))

        patch_to_species = np.asarray(pts, dtype=np.int64)
        patch_to_small   = np.asarray(pss, dtype=np.int64)
        m_patch          = np.asarray(mps, dtype=np.float64)

        pi1 = unique_edge_ids // M
        pi2 = unique_edge_ids % M
        eps_small = (J[pi1[:, None], pi2[None, :]] + J[pi2[:, None], pi1[None, :]])

    else:
        # DIRECTED boundary types: 0..4*M-1
        pts, pss, mps = [], [], []
        for i in range(P):
            bavg = boundary_avg_list[i]
            for pid, cnt in bavg.items():
                pid = int(pid)
                if pid < 0 or pid >= 4 * M:
                    raise ValueError(f"boundary directed patch id out of range: {pid} (expected 0..{4*M-1})")
                pts.append(i)
                pss.append(pid)
                mps.append(float(cnt))

        patch_to_species = np.asarray(pts, dtype=np.int64)

        if not compress_boundary:
            patch_to_species = np.empty(8 * P, dtype=np.int64)
            patch_to_small   = np.empty(8 * P, dtype=np.int64)
            m_patch          = np.ones(8 * P, dtype=np.float64)

            for i in range(P):
                sUL, sUR, sBL, sBR = map(int, plaq_configs[i])
                pUL = patches[sUL]; pUR = patches[sUR]; pBL = patches[sBL]; pBR = patches[sBR]

                b = np.array([
                    0*M + int(pUL[N]),    0*M + int(pUR[N]),
                    1*M + int(pUR[E]),    1*M + int(pBR[E]),
                    2*M + int(pBR[Sdir]), 2*M + int(pBL[Sdir]),
                    3*M + int(pBL[W]),    3*M + int(pUL[W]),
                ], dtype=np.int64)

                patch_to_species[8*i:8*i+8] = i
                patch_to_small[8*i:8*i+8]   = b

        else:
            pts, pss, mps = [], [], []
            for i in range(P):
                sUL, sUR, sBL, sBR = map(int, plaq_configs[i])
                pUL = patches[sUL]; pUR = patches[sUR]; pBL = patches[sBL]; pBR = patches[sBR]

                b = np.array([
                    0*M + int(pUL[N]),    0*M + int(pUR[N]),
                    1*M + int(pUR[E]),    1*M + int(pBR[E]),
                    2*M + int(pBR[Sdir]), 2*M + int(pBL[Sdir]),
                    3*M + int(pBL[W]),    3*M + int(pUL[W]),
                ], dtype=np.int64)

                uniq, cnt = np.unique(b, return_counts=True)
                pts.append(np.full(uniq.shape[0], i, dtype=np.int64))
                pss.append(uniq.astype(np.int64, copy=False))
                mps.append(cnt.astype(np.float64, copy=False))

            patch_to_species = np.concatenate(pts, axis=0) if pts else np.empty(0, dtype=np.int64)
            patch_to_small   = np.concatenate(pss, axis=0) if pss else np.empty(0, dtype=np.int64)
            m_patch          = np.concatenate(mps, axis=0) if mps else np.empty(0, dtype=np.float64)

        eps_small = np.zeros((4 * M, 4 * M), dtype=np.float64)
        eps_small[0*M:1*M, 2*M:3*M] = J
        eps_small[2*M:3*M, 0*M:1*M] = J.T
        eps_small[1*M:2*M, 3*M:4*M] = J
        eps_small[3*M:4*M, 1*M:2*M] = J.T

    if return_class_members:
        return (
            eps_small,
            intra_bonds,
            plaq_to_species,
            m_patch,
            patch_to_species,
            patch_to_small,
            plaq_configs,
            mult_arr,
            plaq_class_members,
        )

    return eps_small, intra_bonds, plaq_to_species, m_patch, patch_to_species, patch_to_small, plaq_configs, mult_arr


def build_cubes_species(
    patches: np.ndarray,
    J: np.ndarray,
    mu: Optional[np.ndarray] = None,
    rot90_species: Optional[np.ndarray] = None,
    *,
    compress_boundary: bool = True,
    use_4_patch: bool = True,
    undirected_edges: bool = False,
    return_class_members: bool = False,
) -> Any:
    """
    Cube-level builder with the same return contract as build_plaquettes_by_species,
    but restricted to boundary-edge canonicalization only.

    This intentionally disables canonicalize_by_edges and always enables
    canonicalize_by_boundary_edges.
    """
    if not use_4_patch:
        raise ValueError("build_cubes_species requires use_4_patch=True for boundary-edge canonicalization.")

    return build_plaquettes_by_species(
        patches=patches,
        J=J,
        mu=mu,
        rot90_species=rot90_species,
        compress_boundary=compress_boundary,
        use_4_patch=use_4_patch,
        undirected_edges=undirected_edges,
        canonicalize_by_edges=False,
        canonicalize_by_boundary_edges=True,
        return_class_members=return_class_members,
    )

def residual_free_energy_simple(
    rho: torch.Tensor,
    beta: float,
    z_assoc: float,
    patch_to_species: torch.Tensor,
    patch_to_small: torch.Tensor,
    eps_small: torch.Tensor,
    m_patch: torch.Tensor,
    e_linear: torch.Tensor,
    delta_small: torch.Tensor,
    mult: torch.Tensor,
    *,
    X0: Optional[torch.Tensor] = None,
    return_X: bool = False,
    max_iter: int = 1000,
    tol: float = 1e-5,
    damp: float = 0.7,
):
    # Keep same convention as the notebooks: beta is unused since J is pre-scaled by 1/T.
    eps = torch.tensor(1e-12, dtype=rho.dtype, device=rho.device)
    A_mix = torch.sum(rho * (torch.log(rho + eps) - torch.log(mult + eps)))
    A_lin = torch.dot(e_linear, rho)

    M = int(eps_small.shape[0])
    phi_p = rho[patch_to_species]
    X = torch.ones_like(phi_p) if X0 is None else X0.clone()
    for _ in range(max_iter):
        W = torch.zeros(M, dtype=rho.dtype, device=rho.device)
        W.index_add_(0, patch_to_small, m_patch * phi_p * X)
        u = delta_small.matmul(W)
        denom = (1.0 + u[patch_to_small]).clamp_min(1e-12)
        X_new = (1.0 / denom).clamp(1e-12, 1.0)
        if torch.max(torch.abs(X_new - X)) < tol:
            X = X_new
            break
        X = damp * X_new + (1.0 - damp) * X

    A_assoc = torch.sum(m_patch * phi_p * (torch.log(X + eps) - 0.5 * X + 0.5))
    f = (A_lin + A_assoc + A_mix) / 7.0
    if return_X:
        return f, X
    return f


# =========================
# Triangular-lattice hexagons
# =========================

def _rot60_species_map(n_species: int, rot60_species: Optional[np.ndarray]) -> np.ndarray:
    """
    rot60_species: permutation of [0..S-1] giving the species after a 60° clockwise rotation.
    """
    if rot60_species is None:
        return np.arange(n_species, dtype=np.int64)

    rot60_species = np.asarray(rot60_species, dtype=np.int64)
    if rot60_species.shape != (n_species,):
        raise ValueError(f"rot60_species must have shape ({n_species},), got {rot60_species.shape}")
    if rot60_species.min() < 0 or rot60_species.max() >= n_species:
        raise ValueError("rot60_species entries must be in [0, n_species).")
    if np.unique(rot60_species).size != n_species:
        raise ValueError("rot60_species must be a permutation (each species appears exactly once).")
    return rot60_species


def _rotate_hex_cfg(
    cfg: Tuple[int, int, int, int, int, int],
    rot60_species: np.ndarray,
    k: int,
) -> Tuple[int, int, int, int, int, int]:
    """
    Rotate a 6-site triangular-lattice hexagon by k quarter-turns of 60° clockwise,
    applying the per-species 60° rotation map to each site species.

    Site order convention used throughout:
      (0,1,2,3,4,5) = (E, NE, NW, W, SW, SE)

    Under a 60° clockwise global rotation, ring sites map as:
      new site i <- old site (i+1) mod 6

    while each species is relabeled by rot60_species.
    """
    out = tuple(int(x) for x in cfg)
    for _ in range(int(k) % 6):
        out = tuple(int(rot60_species[out[(i + 1) % 6]]) for i in range(6))
    return out


def _hex_decode_chunk(start: int, stop: int, n_species: int) -> np.ndarray:
    idx = np.arange(start, stop, dtype=np.int64)
    B = idx.shape[0]
    cfg = np.empty((B, 6), dtype=np.int64)
    tmp = idx.copy()
    for pos in range(5, -1, -1):
        cfg[:, pos] = tmp % n_species
        tmp //= n_species
    return cfg


def _lex_lt_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Vectorized row-wise lexicographic comparison: a < b."""
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch in _lex_lt_rows: {a.shape} vs {b.shape}")
    B, L = a.shape
    take = np.zeros(B, dtype=bool)
    undec = np.ones(B, dtype=bool)
    for j in range(L):
        lt = a[:, j] < b[:, j]
        gt = a[:, j] > b[:, j]
        take |= undec & lt
        undec &= ~(lt | gt)
    return take


def _lexicographic_min_over_rotations(cands: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    cands: (B, K, L)
    Returns:
      best     : (B, L) lexicographically minimal row across K candidates
      best_idx : (B,)   which candidate attained the minimum (first if ties)
    """
    if cands.ndim != 3:
        raise ValueError(f"cands must have ndim=3, got {cands.ndim}")
    B, K, L = cands.shape
    best = cands[:, 0, :].copy()
    best_idx = np.zeros(B, dtype=np.int64)
    for k in range(1, K):
        cand = cands[:, k, :]
        take = _lex_lt_rows(cand, best)
        if np.any(take):
            best[take] = cand[take]
            best_idx[take] = k
    return best, best_idx


def build_hexagons_by_species(
    patches: np.ndarray,
    J: np.ndarray,
    mu: Optional[np.ndarray] = None,
    rot60_species: Optional[np.ndarray] = None,
    *,
    compress_boundary: bool = True,
    use_2_patch: bool = True,
    undirected_edges: bool = False,
    canonicalize_by_boundary_edges: bool = False,
    return_class_members: bool = False,
    chunk_size: int = 200_000,
) -> Any:
    """
    Vectorized 6-site hexagon builder for a triangular lattice.

    Patch-order convention for each species row in `patches`:
      (0,1,2,3,4,5) = (E, NE, NW, W, SW, SE)

    Site-order convention for the 6 hexagon vertices:
      (0,1,2,3,4,5) = (E, NE, NW, W, SW, SE)

    The hexagon is the ring of six nearest neighbors around a central site.
    Internal bonds are the 6 ring bonds between consecutive vertices.

    Boundary-side convention:
      side i is the side between site i and site (i+1) mod 6.
      It carries TWO outward patches, encoded in the fixed order
        (site i patch toward the outer apex, site i+1 patch toward the same outer apex).

      With the above directional convention this becomes
        side i = (
          patches[site i,     direction (i+1) mod 6],
          patches[site i + 1, direction i],
        )

      With this ordering, when two hexagons meet side-to-side, the correct superpatch
      interaction is INDEXWISE:
        (a0, a1) interacts with (b0, b1) as J[a0,b0] + J[a1,b1]

    Returns the same style of tuple as build_plaquettes_by_species:
      eps_small,
      intra_bonds,
      hex_to_species,
      m_patch,
      patch_to_species,
      patch_to_small,
      hex_configs,
      mult_arr
    """
    if not compress_boundary:
        raise ValueError("build_hexagons_by_species currently requires compress_boundary=True")
    if not use_2_patch:
        raise ValueError("build_hexagons_by_species currently requires use_2_patch=True")

    patches = np.asarray(patches, dtype=np.int64)
    J = np.asarray(J, dtype=np.float64)

    Ssp = int(patches.shape[0])
    n_dirs = int(patches.shape[1])
    if n_dirs != 6:
        raise ValueError(f"hexagon builder expects patches.shape[1] == 6, got {n_dirs}")

    M = int(J.shape[0])
    if J.shape != (M, M):
        raise ValueError(f"J must be square, got {J.shape}")
    if patches.max() >= M or patches.min() < 0:
        raise ValueError("patch ids in patches must be within [0, J.shape[0])")

    mu_vec = np.zeros(Ssp, dtype=np.float64) if mu is None else np.asarray(mu, dtype=np.float64)
    if mu_vec.shape != (Ssp,):
        raise ValueError(f"mu must have shape ({Ssp},), got {mu_vec.shape}")

    rot_map = _rot60_species_map(Ssp, rot60_species)
    rot_powers = [np.arange(Ssp, dtype=np.int64)]
    for _ in range(5):
        rot_powers.append(rot_map[rot_powers[-1]])
    rot_powers = np.asarray(rot_powers, dtype=np.int64)  # (6,S)

    # Triangular-lattice ring site coordinates in the SAME order as the site / direction convention:
    # (E, NE, NW, W, SW, SE)
    coords = np.asarray([
        [ 1,  0],
        [ 0,  1],
        [-1,  1],
        [-1,  0],
        [ 0, -1],
        [ 1, -1],
    ], dtype=np.int64)
    vec_to_dir = {tuple(v.tolist()): i for i, v in enumerate(coords)}

    # Internal ring bonds between consecutive sites.
    bond_dir_fwd = np.empty(6, dtype=np.int64)
    bond_dir_bwd = np.empty(6, dtype=np.int64)
    for i in range(6):
        j = (i + 1) % 6
        bond_dir_fwd[i] = vec_to_dir[tuple((coords[j] - coords[i]).tolist())]
        bond_dir_bwd[i] = vec_to_dir[tuple((coords[i] - coords[j]).tolist())]

    def _side_id(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        if undirected_edges:
            lo = np.minimum(a, b)
            hi = np.maximum(a, b)
            return lo * M + hi
        return a * M + b

    total = int(Ssp ** 6)
    global_groups: Dict[Any, Dict[str, Any]] = {}

    for start in range(0, total, int(chunk_size)):
        stop = min(total, start + int(chunk_size))
        cfg = _hex_decode_chunk(start, stop, Ssp)  # (B,6)
        B = cfg.shape[0]
        rows = np.arange(B, dtype=np.int64)

        p = patches[cfg]  # (B,6,6)

        # 6 internal ring bonds
        E = np.zeros(B, dtype=np.float64)
        for i in range(6):
            j = (i + 1) % 6
            E += J[p[:, i, bond_dir_fwd[i]], p[:, j, bond_dir_bwd[i]]]

        mu_sum = mu_vec[cfg].sum(axis=1)
        Eeff = E - mu_sum if canonicalize_by_boundary_edges else E

        species_counts = np.zeros((B, Ssp), dtype=np.float64)
        for col in range(6):
            species_counts[rows, cfg[:, col]] += 1.0

        # 6 boundary sides, each encoded as a 2-patch superpatch id.
        sides = np.empty((B, 6), dtype=np.int64)
        for i in range(6):
            j = (i + 1) % 6
            a = p[:, i, j]  # site i patch toward the outer apex of side i
            b = p[:, j, i]  # site j patch toward the same outer apex
            sides[:, i] = _side_id(a, b)

        if canonicalize_by_boundary_edges:
            side_rots = np.stack([np.roll(sides, -k, axis=1) for k in range(6)], axis=1)  # (B,6,6)
            keys, _ = _lexicographic_min_over_rotations(side_rots)
            boundary_counts = keys
            orbit_sizes = None
        else:
            cfg_rots = np.empty((B, 6, 6), dtype=np.int64)
            base_idx = np.arange(6, dtype=np.int64)
            for k in range(6):
                cfg_rots[:, k, :] = rot_powers[k][cfg[:, (base_idx + k) % 6]]
            keys, _ = _lexicographic_min_over_rotations(cfg_rots)
            boundary_counts = sides
            orbit_sizes = cfg_rots

        uniq_keys, inv = np.unique(keys, axis=0, return_inverse=True)
        G = uniq_keys.shape[0]

        min_e = np.full(G, np.inf, dtype=np.float64)
        np.minimum.at(min_e, inv, Eeff)

        contrib = np.exp(-(Eeff - min_e[inv]))
        w = np.zeros(G, dtype=np.float64)
        np.add.at(w, inv, contrib)

        species_sum = np.zeros((G, Ssp), dtype=np.float64)
        np.add.at(species_sum, inv, species_counts * contrib[:, None])

        # Representative config = lowest-Eeff member in this chunk group.
        order = np.lexsort((Eeff, inv))
        inv_sorted = inv[order]
        first = np.concatenate(([True], inv_sorted[1:] != inv_sorted[:-1]))
        best_cfg = np.empty((G, 6), dtype=np.int64)
        best_cfg[inv_sorted[first]] = cfg[order[first]]

        for g in range(G):
            key_t = tuple(int(x) for x in uniq_keys[g].tolist())
            rec = global_groups.get(key_t)
            if rec is None:
                global_groups[key_t] = {
                    "E0": float(min_e[g]),
                    "w": float(w[g]),
                    "species_sum": species_sum[g].copy(),
                    "best_cfg": tuple(int(x) for x in best_cfg[g].tolist()),
                }
                continue

            oldE = float(rec["E0"])
            newE = float(min_e[g])
            if newE < oldE:
                fac = float(np.exp(-(oldE - newE)))
                rec["w"] *= fac
                rec["species_sum"] *= fac
                rec["w"] += float(w[g])
                rec["species_sum"] += species_sum[g]
                rec["E0"] = newE
                rec["best_cfg"] = tuple(int(x) for x in best_cfg[g].tolist())
            else:
                fac = float(np.exp(-(newE - oldE)))
                rec["w"] += float(w[g]) * fac
                rec["species_sum"] += species_sum[g] * fac

    reps = []
    intra_bonds = []
    mult_arr = []
    hex_to_species_rows = []
    boundary_avg_list = []

    for key_t, rec in global_groups.items():
        w = float(rec["w"])
        if (not np.isfinite(rec["E0"])) or (w <= 0.0):
            continue

        cfg_t = tuple(int(x) for x in rec["best_cfg"])
        reps.append(cfg_t)
        intra_bonds.append(float(rec["E0"]))
        hex_to_species_rows.append(rec["species_sum"] / w)

        uniq_side, cnt_side = np.unique(np.asarray(key_t, dtype=np.int64), return_counts=True)
        boundary_avg_list.append({int(u): float(c) for u, c in zip(uniq_side, cnt_side)})

        if canonicalize_by_boundary_edges:
            mult_arr.append(w)
        else:
            rots = [_rotate_hex_cfg(cfg_t, rot_map, k) for k in range(6)]
            mult_arr.append(float(len(set(rots))))

    hex_configs = np.asarray(reps, dtype=np.int64)
    intra_bonds = np.asarray(intra_bonds, dtype=np.float64)
    hex_to_species = np.asarray(hex_to_species_rows, dtype=np.float64)
    mult_arr = np.asarray(mult_arr, dtype=np.float64)
    P = int(hex_configs.shape[0])

    all_side_ids = sorted({int(t) for bavg in boundary_avg_list for t in bavg.keys()})
    unique_side_ids = np.asarray(all_side_ids, dtype=np.int64)
    side_id_to_idx = {int(sid): idx for idx, sid in enumerate(unique_side_ids.tolist())}

    pts, pss, mps = [], [], []
    for i in range(P):
        bavg = boundary_avg_list[i]
        for sid, cnt in bavg.items():
            pts.append(i)
            pss.append(side_id_to_idx[int(sid)])
            mps.append(float(cnt))

    patch_to_species = np.asarray(pts, dtype=np.int64)
    patch_to_small = np.asarray(pss, dtype=np.int64)
    m_patch = np.asarray(mps, dtype=np.float64)

    pi1 = unique_side_ids // M
    pi2 = unique_side_ids % M
    eps_small = J[pi1[:, None], pi1[None, :]] + J[pi2[:, None], pi2[None, :]]

    if return_class_members:
        class_members = [tuple(sorted(set(_rotate_hex_cfg(tuple(int(x) for x in cfg.tolist()), rot_map, k) for k in range(6))))
                         for cfg in hex_configs]
        return (
            eps_small,
            intra_bonds,
            hex_to_species,
            m_patch,
            patch_to_species,
            patch_to_small,
            hex_configs,
            mult_arr,
            class_members,
        )

    return eps_small, intra_bonds, hex_to_species, m_patch, patch_to_species, patch_to_small, hex_configs, mult_arr


# singular alias for convenience
build_hexagon_species = build_hexagons_by_species
import numpy as np
from typing import Optional, Tuple, Dict, Any



def _rotate_hex7_cfg(
    cfg: Tuple[int, int, int, int, int, int, int],
    rot60_species: np.ndarray,
    k: int,
) -> Tuple[int, int, int, int, int, int, int]:
    """
    Rotate a 7-site compact triangular-lattice hexagon by k steps of 60° clockwise.

    Site-order convention:
      (0,1,2,3,4,5,6) = (C, E, NE, NW, W, SW, SE)

    Under one 60° clockwise rotation:
      - the center stays at the center and is relabeled by rot60_species
      - new ring site i <- old ring site (i+1) mod 6, after species relabeling
    """
    c, *ring = [int(x) for x in cfg]
    for _ in range(int(k) % 6):
        c = int(rot60_species[c])
        ring = [int(rot60_species[ring[(i + 1) % 6]]) for i in range(6)]
    return (c, *ring)


def _hex7_decode_chunk(start: int, stop: int, n_species: int) -> np.ndarray:
    idx = np.arange(start, stop, dtype=np.int64)
    B = idx.shape[0]
    cfg = np.empty((B, 7), dtype=np.int64)
    tmp = idx.copy()
    for pos in range(6, -1, -1):
        cfg[:, pos] = tmp % n_species
        tmp //= n_species
    return cfg


def build_hexagon7_by_species(
    patches: np.ndarray,
    J: np.ndarray,
    mu: Optional[np.ndarray] = None,
    rot60_species: Optional[np.ndarray] = None,
    *,
    compress_boundary: bool = True,
    use_3_patch: bool = True,
    undirected_edges: bool = False,
    canonicalize_by_boundary_edges: bool = True,
    split_center_vacancy: bool = False,
    composition_key: str = "components",
    species_to_component: Any = None,
    vacancy_index: int = -1,
    return_class_members: bool = False,
    return_composition_info: bool = False,
    chunk_size: int = 200_000,
) -> Any:
    """
    Vectorized 7-site compact-hexagon builder for a triangular lattice.

    ``composition_key`` (composition-aware classes)
    -----------------------------------------------
    The six boundary faces are built from the ring sites only, so the centre
    site is invisible to the class key and ring sites contribute only their
    outward patches.  With the plain boundary key, hexagons holding different
    numbers of particles therefore landed in the same class and were
    Boltzmann-averaged into a single fractional composition.  A 60 deg rotation
    is a symmetry of the cluster and may be collapsed; a change of composition
    is not, and must not be.

        "components"  (default) class key = boundary-face signature x count per
                      chemical component, where a component is a set of species
                      closed under ``rot60_species`` (by default exactly the
                      orbits of that map, so the six orientations of one
                      particle collapse while chemically distinct particles and
                      the vacancy stay separate).  Because components are unions
                      of rotation orbits the count vector is C6-invariant and is
                      simply prepended to the canonical key.

        "species"     one component per species; only well defined when
                      ``rot60_species`` does not relabel species.

        "none"        legacy behaviour: boundary-face signature only.

    ``split_center_vacancy`` is the older, partial version of the same idea (it
    separates centre-vacant from centre-occupied but nothing else).  It stays
    supported and is orthogonal: it is a *positional* distinction, whereas
    ``composition_key`` is a compositional one.  With ``composition_key`` on,
    ``split_center_vacancy`` is usually redundant.

    Patch-order convention for each species row in `patches`:
      (0,1,2,3,4,5) = (E, NE, NW, W, SW, SE)

    Cluster site-order convention:
      (0,1,2,3,4,5,6) = (C, E, NE, NW, W, SW, SE)

    Internal bonds:
      - 6 center-to-ring bonds
      - 6 ring bonds between consecutive ring sites
      Total = 12 nearest-neighbor bonds.

    Boundary-face convention:
      There are 6 geometric boundary faces. In cyclic order we encode them as
        (SE, S, SW, NW, N, NE).
      Each face is spanned by 2 adjacent boundary particles and carries 3 exposed
      patch slots total.

      The face-slot convention is generated by 60° rotation from the SE face template
        SE face = (E.SE, SE.E, SE.SE)
      so the full ordered list is
        SE : (E.SE,  SE.E,  SE.SE)
        S  : (SE.SW, SW.SE, SW.SW)
        SW : (SW.W,  W.SW,  W.W)
        NW : (W.NW,  NW.W,  NW.NW)
        N  : (NW.NE, NE.NW, NE.NE)
        NE : (NE.E,  E.NE,  E.E)

      With this rotationally covariant face encoding, opposite faces meet with REVERSED
      slot pairing. For example, the SE face of one cluster contacts the NW face of the
      neighboring cluster as
        E1.SE  <-> NW2.NW
        SE1.E  <-> NW2.W
        SE1.SE <-> W2.NW
      which is exactly the reversed pairing
        (a0, a1, a2) against (b0, b1, b2)
          -> J[a0, b2] + J[a1, b1] + J[a2, b0]

    Boltzmann averaging:
      When canonicalize_by_boundary_edges=True, all raw cluster realizations with the same
      boundary-face signature up to 60° rotation are grouped together, and the resulting
      macrostate uses Boltzmann-averaged species counts within that boundary class.

    Returns the same style of tuple as build_plaquettes_by_species:
      eps_small,
      intra_bonds,
      hex_to_species,
      m_patch,
      patch_to_species,
      patch_to_small,
      hex_configs,
      mult_arr
    """
    if not compress_boundary:
        raise ValueError("build_hexagon7_by_species currently requires compress_boundary=True")
    if not use_3_patch:
        raise ValueError("build_hexagon7_by_species currently requires use_3_patch=True")
    if undirected_edges:
        raise ValueError("build_hexagon7_by_species requires ordered 3-slot boundary faces; undirected_edges=False only")
    if split_center_vacancy and not canonicalize_by_boundary_edges:
        raise ValueError("split_center_vacancy=True requires canonicalize_by_boundary_edges=True")

    composition_key = str(composition_key).strip().lower()
    if composition_key not in COMPOSITION_KEYS:
        raise ValueError(f"composition_key must be one of {COMPOSITION_KEYS}, got {composition_key!r}")
    if not canonicalize_by_boundary_edges:
        # The class is then the full rotation orbit of one microstate, whose
        # composition is already exact.
        composition_key = "none"

    patches = np.asarray(patches, dtype=np.int64)
    J = np.asarray(J, dtype=np.float64)

    Ssp = int(patches.shape[0])
    n_dirs = int(patches.shape[1])
    if n_dirs != 6:
        raise ValueError(f"hexagon7 builder expects patches.shape[1] == 6, got {n_dirs}")

    M = int(J.shape[0])
    if J.shape != (M, M):
        raise ValueError(f"J must be square, got {J.shape}")
    if patches.max() >= M or patches.min() < 0:
        raise ValueError("patch ids in patches must be within [0, J.shape[0])")

    mu_vec = np.zeros(Ssp, dtype=np.float64) if mu is None else np.asarray(mu, dtype=np.float64)
    if mu_vec.shape != (Ssp,):
        raise ValueError(f"mu must have shape ({Ssp},), got {mu_vec.shape}")

    rot_map = _rot60_species_map(Ssp, rot60_species)
    rot_powers = [np.arange(Ssp, dtype=np.int64)]
    for _ in range(5):
        rot_powers.append(rot_map[rot_powers[-1]])
    rot_powers = np.asarray(rot_powers, dtype=np.int64)  # (6,S)

    if composition_key == "none":
        comp_map = np.arange(Ssp, dtype=np.int64)
        n_components = Ssp
    else:
        comp_map, n_components = resolve_component_map(
            Ssp,
            "species" if composition_key == "species" else species_to_component,
            rot_map=rot_map,
            vacancy_index=vacancy_index,
            rot_name="rot60_species",
        )

    # Leading columns of the class key that are NOT boundary faces.
    # Key layout is (composition?, center_flag?, face0..face5).
    n_prefix_cols = int(composition_key != "none") + int(bool(split_center_vacancy))

    # Ring site order = (E, NE, NW, W, SW, SE)
    ring_coords = np.asarray([
        [ 1,  0],
        [ 0,  1],
        [-1,  1],
        [-1,  0],
        [ 0, -1],
        [ 1, -1],
    ], dtype=np.int64)
    vec_to_dir = {tuple(v.tolist()): i for i, v in enumerate(ring_coords)}

    # Consecutive ring bonds.
    ring_fwd = np.empty(6, dtype=np.int64)
    ring_bwd = np.empty(6, dtype=np.int64)
    for i in range(6):
        j = (i + 1) % 6
        ring_fwd[i] = vec_to_dir[tuple((ring_coords[j] - ring_coords[i]).tolist())]
        ring_bwd[i] = vec_to_dir[tuple((ring_coords[i] - ring_coords[j]).tolist())]

    # Boundary faces in cyclic order (SE, S, SW, NW, N, NE), each with 3 slots.
    # Each row lists ((site0, dir0), (site1, dir1), (site2, dir2)).
    # Ring site / direction indices both use the order (E, NE, NW, W, SW, SE).
    face_specs = np.asarray([
        [0, 5, 5, 0, 5, 5],  # SE : (E.SE,  SE.E,  SE.SE)
        [5, 4, 4, 5, 4, 4],  # S  : (SE.SW, SW.SE, SW.SW)
        [4, 3, 3, 4, 3, 3],  # SW : (SW.W,  W.SW,  W.W)
        [3, 2, 2, 3, 2, 2],  # NW : (W.NW,  NW.W,  NW.NW)
        [2, 1, 1, 2, 1, 1],  # N  : (NW.NE, NE.NW, NE.NE)
        [1, 0, 0, 1, 0, 0],  # NE : (NE.E,  E.NE,  E.E)
    ], dtype=np.int64)

    def _face_id(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
        return (a * M + b) * M + c

    def _extract_faces_from_cfg(cfg_block: np.ndarray) -> np.ndarray:
        ring_species_block = cfg_block[:, 1:]  # (B,6) in order (E, NE, NW, W, SW, SE)
        Bblk = cfg_block.shape[0]
        faces_block = np.empty((Bblk, 6), dtype=np.int64)
        for d in range(6):
            s0, p0, s1, p1, s2, p2 = face_specs[d]
            a = patches[ring_species_block[:, s0], p0]
            b = patches[ring_species_block[:, s1], p1]
            c = patches[ring_species_block[:, s2], p2]
            faces_block[:, d] = _face_id(a, b, c)
        return faces_block

    total = int(Ssp ** 7)
    global_groups: Dict[Any, Dict[str, Any]] = {}

    for start in range(0, total, int(chunk_size)):
        stop = min(total, start + int(chunk_size))
        cfg = _hex7_decode_chunk(start, stop, Ssp)  # (B,7)
        B = cfg.shape[0]
        rows = np.arange(B, dtype=np.int64)

        p = patches[cfg]  # (B,7,6)

        # 6 center-ring bonds.
        E = np.zeros(B, dtype=np.float64)
        for d in range(6):
            ring_site = 1 + d
            opp = (d + 3) % 6
            E += J[p[:, 0, d], p[:, ring_site, opp]]

        # 6 ring-ring bonds.
        for i in range(6):
            j = (i + 1) % 6
            si = 1 + i
            sj = 1 + j
            E += J[p[:, si, ring_fwd[i]], p[:, sj, ring_bwd[i]]]

        mu_sum = mu_vec[cfg].sum(axis=1)
        Eeff = E - mu_sum if canonicalize_by_boundary_edges else E

        species_counts = np.zeros((B, Ssp), dtype=np.float64)
        for col in range(7):
            species_counts[rows, cfg[:, col]] += 1.0

        # 6 boundary faces, each a 3-slot superpatch built from 2 boundary particles.
        faces = _extract_faces_from_cfg(cfg)

        if canonicalize_by_boundary_edges:
            cfg_rots = np.empty((B, 6, 7), dtype=np.int64)
            base_idx = np.arange(6, dtype=np.int64)
            for k in range(6):
                cfg_rots[:, k, 0] = rot_powers[k][cfg[:, 0]]
                cfg_rots[:, k, 1:] = rot_powers[k][cfg[:, 1 + ((base_idx + k) % 6)]]
            face_rots = np.empty((B, 6, 6), dtype=np.int64)
            for k in range(6):
                face_rots[:, k, :] = _extract_faces_from_cfg(cfg_rots[:, k, :])
            keys, _ = _lexicographic_min_over_rotations(face_rots)
            if split_center_vacancy:
                # Vacancy is always the last species (Ssp-1) and maps to itself
                # under rotation, so this flag is rotation-invariant.
                center_flag = (cfg[:, 0] == Ssp - 1).astype(np.int64).reshape(-1, 1)
                keys = np.column_stack([center_flag, keys])
            if composition_key != "none":
                # Components are unions of rot60 orbits, so this code is the
                # same for all six rotated images of a hexagon and can be
                # prepended to the canonical key with no further bookkeeping.
                comp_codes, _ = composition_codes_bulk(
                    species_counts.astype(np.int64), comp_map, n_components, 7)
                keys = np.column_stack([comp_codes.reshape(-1, 1), keys])
        else:
            cfg_rots = np.empty((B, 6, 7), dtype=np.int64)
            base_idx = np.arange(6, dtype=np.int64)
            for k in range(6):
                cfg_rots[:, k, 0] = rot_powers[k][cfg[:, 0]]
                cfg_rots[:, k, 1:] = rot_powers[k][cfg[:, 1 + ((base_idx + k) % 6)]]
            keys, _ = _lexicographic_min_over_rotations(cfg_rots)

        uniq_keys, inv = np.unique(keys, axis=0, return_inverse=True)
        G = uniq_keys.shape[0]

        min_e = np.full(G, np.inf, dtype=np.float64)
        np.minimum.at(min_e, inv, Eeff)

        contrib = np.exp(-(Eeff - min_e[inv]))
        w = np.zeros(G, dtype=np.float64)
        np.add.at(w, inv, contrib)

        species_sum = np.zeros((G, Ssp), dtype=np.float64)
        np.add.at(species_sum, inv, species_counts * contrib[:, None])

        # Representative config = lowest-Eeff member in this chunk group.
        order = np.lexsort((Eeff, inv))
        inv_sorted = inv[order]
        first = np.concatenate(([True], inv_sorted[1:] != inv_sorted[:-1]))
        best_cfg = np.empty((G, 7), dtype=np.int64)
        best_cfg[inv_sorted[first]] = cfg[order[first]]

        for g in range(G):
            key_t = tuple(int(x) for x in uniq_keys[g].tolist())
            rec = global_groups.get(key_t)
            if rec is None:
                global_groups[key_t] = {
                    "E0": float(min_e[g]),
                    "w": float(w[g]),
                    "species_sum": species_sum[g].copy(),
                    "best_cfg": tuple(int(x) for x in best_cfg[g].tolist()),
                }
                continue

            oldE = float(rec["E0"])
            newE = float(min_e[g])
            if newE < oldE:
                fac = float(np.exp(-(oldE - newE)))
                rec["w"] *= fac
                rec["species_sum"] *= fac
                rec["w"] += float(w[g])
                rec["species_sum"] += species_sum[g]
                rec["E0"] = newE
                rec["best_cfg"] = tuple(int(x) for x in best_cfg[g].tolist())
            else:
                fac = float(np.exp(-(newE - oldE)))
                rec["w"] += float(w[g]) * fac
                rec["species_sum"] += species_sum[g] * fac

    reps = []
    intra_bonds = []
    mult_arr = []
    hex_to_species_rows = []
    boundary_avg_list = []

    for key_t, rec in global_groups.items():
        w = float(rec["w"])
        if (not np.isfinite(rec["E0"])) or (w <= 0.0):
            continue

        cfg_t = tuple(int(x) for x in rec["best_cfg"])
        reps.append(cfg_t)
        intra_bonds.append(float(rec["E0"]))
        hex_to_species_rows.append(rec["species_sum"] / w)

        # Key layout is (composition?, center_flag?, face0..face5); the face
        # information lives in the last 6 elements only.
        face_key = key_t[n_prefix_cols:] if n_prefix_cols else key_t
        uniq_face, cnt_face = np.unique(np.asarray(face_key, dtype=np.int64), return_counts=True)
        boundary_avg_list.append({int(u): float(c) for u, c in zip(uniq_face, cnt_face)})

        if canonicalize_by_boundary_edges:
            mult_arr.append(1)
        else:
            rots = [_rotate_hex7_cfg(cfg_t, rot_map, k) for k in range(6)]
            mult_arr.append(float(len(set(rots))))

    hex_configs = np.asarray(reps, dtype=np.int64)
    intra_bonds = np.asarray(intra_bonds, dtype=np.float64)
    hex_to_species = np.asarray(hex_to_species_rows, dtype=np.float64)
    mult_arr = np.asarray(mult_arr, dtype=np.float64)
    P = int(hex_configs.shape[0])

    all_face_ids = sorted({int(t) for bavg in boundary_avg_list for t in bavg.keys()})
    unique_face_ids = np.asarray(all_face_ids, dtype=np.int64)
    face_id_to_idx = {int(fid): idx for idx, fid in enumerate(unique_face_ids.tolist())}

    pts, pss, mps = [], [], []
    for i in range(P):
        bavg = boundary_avg_list[i]
        for fid, cnt in bavg.items():
            pts.append(i)
            pss.append(face_id_to_idx[int(fid)])
            mps.append(float(cnt))

    patch_to_species = np.asarray(pts, dtype=np.int64)
    patch_to_small = np.asarray(pss, dtype=np.int64)
    m_patch = np.asarray(mps, dtype=np.float64)

    # Decode 3-slot face ids and build the REVERSED slot pairing matrix.
    fi0 = unique_face_ids // (M * M)
    rem = unique_face_ids % (M * M)
    fi1 = rem // M
    fi2 = rem % M
    eps_small = (
        J[fi0[:, None], fi2[None, :]]
        + J[fi1[:, None], fi1[None, :]]
        + J[fi2[:, None], fi0[None, :]]
    )

    out = (eps_small, intra_bonds, hex_to_species, m_patch,
           patch_to_species, patch_to_small, hex_configs, mult_arr)

    if return_class_members:
        class_members = [tuple(sorted(set(_rotate_hex7_cfg(tuple(int(x) for x in cfg.tolist()), rot_map, k) for k in range(6))))
                         for cfg in hex_configs]
        out = out + (class_members,)

    if return_composition_info:
        comp_counts = np.zeros((P, n_components), dtype=np.float64)
        np.add.at(comp_counts.T, comp_map, hex_to_species.T)
        resid = np.abs(comp_counts - np.rint(comp_counts))
        out = out + ({
            "composition_key": composition_key,
            "comp_map": comp_map,
            "n_components": n_components,
            "components": describe_components(comp_map, n_components),
            "n_classes": P,
            "component_counts": comp_counts,
            "max_noninteger_residual": float(resid.max()) if resid.size else 0.0,
        },)

    return out


# convenient aliases
build_hexagons7_by_species = build_hexagon7_by_species
build_compact_hexagon_by_species = build_hexagon7_by_species


# ---------------------------------------------------------------------------
# Cached builder: factor geometry (done once) from energy (done per J)
# ---------------------------------------------------------------------------

def build_hexagon7_geometry_cache(
    patches: np.ndarray,
    rot60_species: Optional[np.ndarray] = None,
    *,
    canonicalize_by_boundary_edges: bool = True,
    split_center_vacancy: bool = False,
    composition_key: str = "components",
    species_to_component: Any = None,
    vacancy_index: int = -1,
    chunk_size: int = 200_000,
) -> dict:
    """
    Precompute all J-independent geometry for the 7-site hexagon builder.

    Returns a cache dict that can be passed to ``hexagon7_from_cache(cache, J, mu)``
    to quickly recompute energies for a new J matrix without re-enumerating or
    re-canonicalizing configurations.

    Parameters
    ----------
    split_center_vacancy : bool
        Only active when ``canonicalize_by_boundary_edges=True``.
        When True, hexagons that share the same boundary-face signature are
        kept in *separate* groups depending on whether their **center site**
        holds the vacancy species (last species, index Ssp-1) or a particle.
        This prevents Boltzmann-averaging across the two distinct physical
        situations (solvent-in-center vs particle-in-center) that happen to
        expose the same boundary.
    """
    if split_center_vacancy and not canonicalize_by_boundary_edges:
        raise ValueError("split_center_vacancy=True requires canonicalize_by_boundary_edges=True")
    composition_key = str(composition_key).strip().lower()
    if composition_key not in COMPOSITION_KEYS:
        raise ValueError(f"composition_key must be one of {COMPOSITION_KEYS}, got {composition_key!r}")
    if not canonicalize_by_boundary_edges:
        composition_key = "none"
    patches = np.asarray(patches, dtype=np.int64)
    Ssp = int(patches.shape[0])
    n_dirs = int(patches.shape[1])
    if n_dirs != 6:
        raise ValueError(f"hexagon7 builder expects patches.shape[1] == 6, got {n_dirs}")
    M = int(patches.max()) + 1

    rot_map = _rot60_species_map(Ssp, rot60_species)
    rot_powers = [np.arange(Ssp, dtype=np.int64)]
    for _ in range(5):
        rot_powers.append(rot_map[rot_powers[-1]])
    rot_powers = np.asarray(rot_powers, dtype=np.int64)

    if composition_key == "none":
        comp_map = np.arange(Ssp, dtype=np.int64)
        n_components = Ssp
    else:
        comp_map, n_components = resolve_component_map(
            Ssp,
            "species" if composition_key == "species" else species_to_component,
            rot_map=rot_map,
            vacancy_index=vacancy_index,
            rot_name="rot60_species",
        )

    ring_coords = np.asarray([
        [ 1,  0], [ 0,  1], [-1,  1], [-1,  0], [ 0, -1], [ 1, -1],
    ], dtype=np.int64)
    vec_to_dir = {tuple(v.tolist()): i for i, v in enumerate(ring_coords)}
    ring_fwd = np.empty(6, dtype=np.int64)
    ring_bwd = np.empty(6, dtype=np.int64)
    for i in range(6):
        j = (i + 1) % 6
        ring_fwd[i] = vec_to_dir[tuple((ring_coords[j] - ring_coords[i]).tolist())]
        ring_bwd[i] = vec_to_dir[tuple((ring_coords[i] - ring_coords[j]).tolist())]

    face_specs = np.asarray([
        [0, 5, 5, 0, 5, 5],
        [5, 4, 4, 5, 4, 4],
        [4, 3, 3, 4, 3, 3],
        [3, 2, 2, 3, 2, 2],
        [2, 1, 1, 2, 1, 1],
        [1, 0, 0, 1, 0, 0],
    ], dtype=np.int64)

    def _face_id(a, b, c):
        return (a * M + b) * M + c

    def _extract_faces(cfg_block):
        ring = cfg_block[:, 1:]
        Bblk = cfg_block.shape[0]
        faces = np.empty((Bblk, 6), dtype=np.int64)
        for d in range(6):
            s0, p0, s1, p1, s2, p2 = face_specs[d]
            faces[:, d] = _face_id(
                patches[ring[:, s0], p0],
                patches[ring[:, s1], p1],
                patches[ring[:, s2], p2],
            )
        return faces

    total = int(Ssp ** 7)

    all_bond_a = []
    all_bond_b = []
    all_species_counts = []
    all_keys = []
    all_cfg = []

    for start in range(0, total, int(chunk_size)):
        stop = min(total, start + int(chunk_size))
        cfg = _hex7_decode_chunk(start, stop, Ssp)
        B = cfg.shape[0]
        p = patches[cfg]

        bond_a = np.empty((B, 12), dtype=np.int64)
        bond_b = np.empty((B, 12), dtype=np.int64)
        for d in range(6):
            bond_a[:, d] = p[:, 0, d]
            bond_b[:, d] = p[:, 1 + d, (d + 3) % 6]
        for i in range(6):
            j = (i + 1) % 6
            bond_a[:, 6 + i] = p[:, 1 + i, ring_fwd[i]]
            bond_b[:, 6 + i] = p[:, 1 + j, ring_bwd[i]]

        species_counts = np.zeros((B, Ssp), dtype=np.float64)
        for col in range(7):
            species_counts[np.arange(B), cfg[:, col]] += 1.0

        base_idx = np.arange(6, dtype=np.int64)
        if canonicalize_by_boundary_edges:
            cfg_rots = np.empty((B, 6, 7), dtype=np.int64)
            for k in range(6):
                cfg_rots[:, k, 0] = rot_powers[k][cfg[:, 0]]
                cfg_rots[:, k, 1:] = rot_powers[k][cfg[:, 1 + ((base_idx + k) % 6)]]
            face_rots = np.empty((B, 6, 6), dtype=np.int64)
            for k in range(6):
                face_rots[:, k, :] = _extract_faces(cfg_rots[:, k, :])
            keys, _ = _lexicographic_min_over_rotations(face_rots)
            if split_center_vacancy:
                # Vacancy (Ssp-1) maps to itself under rot60_species, so the
                # center-vacancy flag is rotation-invariant and safe to prepend.
                center_flag = (cfg[:, 0] == Ssp - 1).astype(np.int64).reshape(-1, 1)
                keys = np.column_stack([center_flag, keys])
            if composition_key != "none":
                # Chemical-component counts: rotation-invariant, so prepending
                # them separates classes that differ in composition while
                # leaving the C6 orientational collapse untouched.
                comp_codes, _ = composition_codes_bulk(
                    species_counts.astype(np.int64), comp_map, n_components, 7)
                keys = np.column_stack([comp_codes.reshape(-1, 1), keys])
        else:
            cfg_rots = np.empty((B, 6, 7), dtype=np.int64)
            for k in range(6):
                cfg_rots[:, k, 0] = rot_powers[k][cfg[:, 0]]
                cfg_rots[:, k, 1:] = rot_powers[k][cfg[:, 1 + ((base_idx + k) % 6)]]
            keys, _ = _lexicographic_min_over_rotations(cfg_rots)

        all_bond_a.append(bond_a)
        all_bond_b.append(bond_b)
        all_species_counts.append(species_counts)
        all_keys.append(keys)
        all_cfg.append(cfg)

    bond_a_all = np.concatenate(all_bond_a)
    bond_b_all = np.concatenate(all_bond_b)
    species_all = np.concatenate(all_species_counts)
    keys_all = np.concatenate(all_keys)
    cfg_all = np.concatenate(all_cfg)

    uniq_keys, group = np.unique(keys_all, axis=0, return_inverse=True)
    n_groups = uniq_keys.shape[0]

    # Boundary face info per group (J-independent).
    # When canonicalize_by_boundary_edges the key layout is
    # (composition?, center_flag?, face0..face5) — face info lives in the
    # last 6 elements only.
    face_col_start = int(composition_key != "none") + int(bool(split_center_vacancy))
    all_face_ids_set: set = set()
    group_face_info: list = [None] * n_groups
    for g in range(n_groups):
        key_arr = uniq_keys[g, face_col_start:].astype(np.int64)
        uniq_face, cnt = np.unique(key_arr, return_counts=True)
        bavg = {int(u): float(c) for u, c in zip(uniq_face, cnt)}
        group_face_info[g] = bavg
        all_face_ids_set.update(bavg.keys())

    unique_face_ids = np.asarray(sorted(all_face_ids_set), dtype=np.int64)
    face_id_to_idx = {int(fid): idx for idx, fid in enumerate(unique_face_ids.tolist())}

    pts, pss, mps = [], [], []
    for i in range(n_groups):
        for fid, cnt in group_face_info[i].items():
            pts.append(i)
            pss.append(face_id_to_idx[int(fid)])
            mps.append(float(cnt))

    patch_to_species = np.asarray(pts, dtype=np.int64)
    patch_to_small = np.asarray(pss, dtype=np.int64)
    m_patch = np.asarray(mps, dtype=np.float64)

    fi0 = unique_face_ids // (M * M)
    rem = unique_face_ids % (M * M)
    fi1 = rem // M
    fi2 = rem % M

    return {
        "bond_a": bond_a_all,
        "bond_b": bond_b_all,
        "species_counts": species_all,
        "cfg": cfg_all,
        "group": group,
        "n_groups": n_groups,
        "Ssp": Ssp,
        "M": M,
        "patches": patches.copy(),
        "canonicalize_by_boundary_edges": canonicalize_by_boundary_edges,
        "split_center_vacancy": split_center_vacancy,
        "composition_key": composition_key,
        "comp_map": comp_map,
        "n_components": n_components,
        "patch_to_species": patch_to_species,
        "patch_to_small": patch_to_small,
        "m_patch": m_patch,
        "unique_face_ids": unique_face_ids,
        "fi0": fi0,
        "fi1": fi1,
        "fi2": fi2,
    }


def hexagon7_from_cache(
    cache: dict,
    J: np.ndarray,
    mu: Optional[np.ndarray] = None,
):
    """
    Recompute energy-dependent outputs from a geometry cache and a new J matrix.

    Returns the same tuple as ``build_hexagon7_by_species``:
      (eps_small, intra_bonds, hex_to_species, m_patch,
       patch_to_species, patch_to_small, hex_configs, mult_arr)
    """
    J = np.asarray(J, dtype=np.float64)
    bond_a = cache["bond_a"]
    bond_b = cache["bond_b"]
    species_counts = cache["species_counts"]
    cfg = cache["cfg"]
    group = cache["group"]
    n_groups = cache["n_groups"]
    Ssp = cache["Ssp"]

    mu_vec = np.zeros(Ssp, dtype=np.float64) if mu is None else np.asarray(mu, dtype=np.float64)

    E = J[bond_a, bond_b].sum(axis=1)
    mu_sum = mu_vec[cfg].sum(axis=1)
    Eeff = E - mu_sum

    min_e = np.full(n_groups, np.inf, dtype=np.float64)
    np.minimum.at(min_e, group, Eeff)

    contrib = np.exp(-(Eeff - min_e[group]))
    w = np.zeros(n_groups, dtype=np.float64)
    np.add.at(w, group, contrib)

    species_sum = np.zeros((n_groups, Ssp), dtype=np.float64)
    np.add.at(species_sum, group, species_counts * contrib[:, None])
    hex_to_species = species_sum / w[:, None]

    intra_bonds = min_e.copy()
    mult_arr = np.ones_like(w)

    # Representative config per group (lowest Eeff).
    order = np.lexsort((Eeff, group))
    grp_sorted = group[order]
    first_mask = np.concatenate(([True], grp_sorted[1:] != grp_sorted[:-1]))
    hex_configs = np.empty((n_groups, 7), dtype=np.int64)
    hex_configs[grp_sorted[first_mask]] = cfg[order[first_mask]]

    fi0 = cache["fi0"]
    fi1 = cache["fi1"]
    fi2 = cache["fi2"]
    eps_small = (
        J[fi0[:, None], fi2[None, :]]
        + J[fi1[:, None], fi1[None, :]]
        + J[fi2[:, None], fi0[None, :]]
    )

    return (
        eps_small,
        intra_bonds,
        hex_to_species,
        cache["m_patch"].copy(),
        cache["patch_to_species"].copy(),
        cache["patch_to_small"].copy(),
        hex_configs,
        mult_arr,
    )


def prune_hexagon7_cache(
    cache: dict,
    J_refs,
    mu: Optional[np.ndarray] = None,
    threshold: float = 1e-10,
) -> dict:
    """
    Remove groups whose Boltzmann weight is negligible at ALL reference J
    matrices. This permanently reduces P, speeding up every subsequent
    ``hexagon7_from_cache`` call and the optimizer.

    Parameters
    ----------
    cache : dict from ``build_hexagon7_geometry_cache``
    J_refs : list of J matrices (or a single J matrix)
        Reference interaction matrices spanning the scan range.
    mu : optional chemical potential vector
    threshold : float
        A group is kept if its relative weight ``w[g] / w.max()`` exceeds
        this value at any reference J.

    Returns
    -------
    pruned_cache : dict with the same structure, fewer configs and groups.
    """
    if not isinstance(J_refs, (list, tuple)):
        J_refs = [J_refs]

    bond_a = cache["bond_a"]
    bond_b = cache["bond_b"]
    group = cache["group"]
    n_groups = cache["n_groups"]
    Ssp = cache["Ssp"]
    cfg = cache["cfg"]
    mu_vec = np.zeros(Ssp, dtype=np.float64) if mu is None else np.asarray(mu, dtype=np.float64)

    keep_group = np.zeros(n_groups, dtype=bool)

    for J in J_refs:
        J = np.asarray(J, dtype=np.float64)
        E = J[bond_a, bond_b].sum(axis=1)
        mu_sum = mu_vec[cfg].sum(axis=1) if mu is not None else 0.0
        Eeff = E - mu_sum

        min_e = np.full(n_groups, np.inf, dtype=np.float64)
        np.minimum.at(min_e, group, Eeff)

        contrib = np.exp(-(Eeff - min_e[group]))
        w = np.zeros(n_groups, dtype=np.float64)
        np.add.at(w, group, contrib)

        keep_group |= (w > threshold * w.max())

    n_keep = int(keep_group.sum())
    if n_keep == n_groups:
        return cache

    old_to_new = np.full(n_groups, -1, dtype=np.int64)
    old_to_new[keep_group] = np.arange(n_keep, dtype=np.int64)

    config_mask = keep_group[group]
    pts_mask = keep_group[cache["patch_to_species"]]

    used_small = set(cache["patch_to_small"][pts_mask].tolist())
    old_face = cache["unique_face_ids"]
    face_keep = np.array([i in used_small for i in range(len(old_face))], dtype=bool)
    face_old_to_new = np.full(len(old_face), -1, dtype=np.int64)
    face_old_to_new[face_keep] = np.arange(int(face_keep.sum()), dtype=np.int64)

    new_unique_face_ids = old_face[face_keep]
    M = cache["M"]
    new_fi0 = new_unique_face_ids // (M * M)
    rem = new_unique_face_ids % (M * M)
    new_fi1 = rem // M
    new_fi2 = rem % M

    return {
        "bond_a": bond_a[config_mask],
        "bond_b": bond_b[config_mask],
        "species_counts": cache["species_counts"][config_mask],
        "cfg": cfg[config_mask],
        "group": old_to_new[group[config_mask]],
        "n_groups": n_keep,
        "Ssp": Ssp,
        "M": M,
        "patches": cache["patches"],
        "canonicalize_by_boundary_edges": cache["canonicalize_by_boundary_edges"],
        "split_center_vacancy": cache.get("split_center_vacancy", False),
        "patch_to_species": old_to_new[cache["patch_to_species"][pts_mask]],
        "patch_to_small": face_old_to_new[cache["patch_to_small"][pts_mask]],
        "m_patch": cache["m_patch"][pts_mask],
        "unique_face_ids": new_unique_face_ids,
        "fi0": new_fi0,
        "fi1": new_fi1,
        "fi2": new_fi2,
    }


# ---------------------------------------------------------------------------
# Numba-accelerated reduced solver
# ---------------------------------------------------------------------------

def build_S_matrix(patch_to_species, patch_to_small, m_patch, P, F):
    """Build the (P, F) aggregation matrix S[i, s] = Σ_{p: pts[p]=i, pss[p]=s} m_p."""
    S = np.zeros((P, F), dtype=np.float64)
    for p in range(len(patch_to_species)):
        S[patch_to_species[p], patch_to_small[p]] += m_patch[p]
    return np.ascontiguousarray(S)


@_njit
def _eval_residual_reduced(mu, W, phi, S, delta, A, e, lm):
    """
    Evaluate the 16-dim KKT residual for the reduced (mu, W) formulation.

    Uses the EXACT association gradient via the implicit function theorem
    on the Wertheim fixed-point, computed at face-type level (F×F).
    """
    Fd = W.shape[0]
    P = A.shape[0]

    # --- X and h from W ---
    u = delta @ W
    X = np.empty(Fd)
    hv = np.empty(Fd)
    for s in range(Fd):
        xs = 1.0 / max(1.0 + u[s], 1e-12)
        xs = max(min(xs, 1.0), 1e-12)
        X[s] = xs
        hv[s] = np.log(max(xs, 1e-300)) - 0.5 * xs + 0.5

    # --- First pass: approximate rho (envelope term only) ---
    g0 = S @ hv
    mx = -1e300
    lr = np.empty(P)
    for i in range(P):
        v = 7.0 * mu * A[i] - e[i] - g0[i] + lm[i]
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

    # phi_rho = S^T @ rho  (F-dim)
    pr = np.zeros(Fd)
    for i in range(P):
        ri = rho[i]
        for s in range(Fd):
            pr[s] += S[i, s] * ri

    # --- Implicit derivative correction (F×F operations) ---
    # K_{ss'} = X_s^2 * delta_{ss'} * phi_rho_{s'}
    K = np.empty((Fd, Fd))
    for s in range(Fd):
        xs2 = X[s] * X[s]
        for sp in range(Fd):
            K[s, sp] = xs2 * delta[s, sp] * pr[sp]

    # dA/dX_s = phi_rho_s * (1/X_s - 0.5)
    dAdX = np.empty(Fd)
    for s in range(Fd):
        dAdX[s] = pr[s] * (1.0 / max(X[s], 1e-12) - 0.5)

    # Solve (I + K^T) alpha = dAdX
    IpKt = np.empty((Fd, Fd))
    for s in range(Fd):
        for sp in range(Fd):
            IpKt[s, sp] = K[sp, s]
        IpKt[s, s] += 1.0
    alpha = np.linalg.solve(IpKt, dAdX)

    # beta = -delta^T @ (alpha * X^2)
    aX2 = np.empty(Fd)
    for s in range(Fd):
        aX2[s] = alpha[s] * X[s] * X[s]
    beta = np.empty(Fd)
    for s in range(Fd):
        v = 0.0
        for sp in range(Fd):
            v += delta[sp, s] * aX2[sp]
        beta[s] = -v

    # Corrected association gradient: h + beta * X
    hv_corr = np.empty(Fd)
    for s in range(Fd):
        hv_corr[s] = hv[s] + beta[s] * X[s]

    # --- Second pass: corrected rho ---
    g = S @ hv_corr
    mx = -1e300
    for i in range(P):
        v = 7.0 * mu * A[i] - e[i] - g[i] + lm[i]
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

    # Recompute phi_rho with corrected rho
    for s in range(Fd):
        pr[s] = 0.0
    for i in range(P):
        ri = rho[i]
        for s in range(Fd):
            pr[s] += S[i, s] * ri

    # --- Residuals ---
    R = np.empty(1 + Fd)
    R[0] = -phi
    for i in range(P):
        R[0] += A[i] * rho[i]
    for s in range(Fd):
        R[1 + s] = W[s] - X[s] * pr[s]
    return R, rho, X, pr


@_njit
def _find_initial_mu(phi_target, A, lm, e, S, delta, W):
    """
    Find mu such that the density constraint phi(mu) = phi_target is approximately
    satisfied, using Newton-bisection on the 1D exponential family.
    Accounts for current W (association state) via the g vector.
    """
    P = A.shape[0]
    Fd = S.shape[1]

    # Compute g from W
    u = delta @ W
    hv = np.empty(Fd)
    for s in range(Fd):
        xs = 1.0 / max(1.0 + u[s], 1e-12)
        xs = max(min(xs, 1.0), 1e-12)
        hv[s] = np.log(max(xs, 1e-300)) - 0.5 * xs + 0.5
    g = S @ hv

    # Newton on phi(mu) = sum_i A_i * softmax_i(7*mu*A_i - e_i - g_i + lm_i)
    mu = 0.0
    mu_lo, mu_hi = -200.0, 200.0

    for _ in range(100):
        mx = -1e300
        for i in range(P):
            v = 7.0 * mu * A[i] - e[i] - g[i] + lm[i]
            if v > mx:
                mx = v
        Z = 0.0
        phi_mu = 0.0
        dphi = 0.0
        for i in range(P):
            v = np.exp(7.0 * mu * A[i] - e[i] - g[i] + lm[i] - mx)
            Z += v
        inv_Z = 1.0 / Z
        for i in range(P):
            ri = np.exp(7.0 * mu * A[i] - e[i] - g[i] + lm[i] - mx) * inv_Z
            phi_mu += A[i] * ri
            dphi += 7.0 * A[i] * A[i] * ri
        dphi -= 7.0 * phi_mu * phi_mu  # dphi/dmu = 7 * Var_rho(A)

        err = phi_mu - phi_target
        if abs(err) < 1e-12:
            break

        # Bisection bounds
        if err > 0:
            mu_hi = mu
        else:
            mu_lo = mu

        # Newton step with bisection fallback
        if abs(dphi) > 1e-30:
            mu_new = mu - err / dphi
            if mu_new < mu_lo or mu_new > mu_hi:
                mu_new = 0.5 * (mu_lo + mu_hi)
        else:
            mu_new = 0.5 * (mu_lo + mu_hi)
        mu = mu_new

    return mu


@_njit
def _solve_single_phi_reduced(phi, mu0, W0, S, delta, A, e, lm,
                               max_newton=50, tol=1e-10):
    """Solve for the free-energy-minimizing rho at a single phi using reduced Newton."""
    Fd = W0.shape[0]
    n_p = 1 + Fd
    h_fd = 1e-7

    # Initial (mu, W) from caller (warm start)
    mu = mu0
    W = W0.copy()

    # Check if warm-start is reasonable; if not, find better mu
    R0, _, _, _ = _eval_residual_reduced(mu, W, phi, S, delta, A, e, lm)
    rn0 = 0.0
    for k in range(n_p):
        rn0 += R0[k] * R0[k]
    rn0 = np.sqrt(rn0)
    if rn0 > 0.1:
        mu = _find_initial_mu(phi, A, lm, e, S, delta, W)

    for it in range(max_newton):
        R, rho, X, pr = _eval_residual_reduced(mu, W, phi, S, delta, A, e, lm)
        rn = 0.0
        for k in range(n_p):
            rn += R[k] * R[k]
        rn = np.sqrt(rn)
        if rn < tol:
            break

        Jac = np.empty((n_p, n_p))
        Rp, _, _, _ = _eval_residual_reduced(mu + h_fd, W, phi, S, delta, A, e, lm)
        for k in range(n_p):
            Jac[k, 0] = (Rp[k] - R[k]) / h_fd
        for j in range(Fd):
            Wp = W.copy()
            Wp[j] += h_fd
            Rp2, _, _, _ = _eval_residual_reduced(mu, Wp, phi, S, delta, A, e, lm)
            for k in range(n_p):
                Jac[k, 1 + j] = (Rp2[k] - R[k]) / h_fd

        for k in range(n_p):
            Jac[k, k] += 1e-12

        dp = np.linalg.solve(Jac, -R)
        for k in range(n_p):
            dp[k] = max(min(dp[k], 10.0), -10.0)

        alpha = 1.0
        best_rn_ls = rn
        for _ in range(20):
            mn = mu + alpha * dp[0]
            Wn = W + alpha * dp[1:]
            Rn, _, _, _ = _eval_residual_reduced(mn, Wn, phi, S, delta, A, e, lm)
            rn_ls = 0.0
            for k in range(n_p):
                rn_ls += Rn[k] * Rn[k]
            rn_ls = np.sqrt(rn_ls)
            if rn_ls < best_rn_ls:
                break
            alpha *= 0.5

        mu += alpha * dp[0]
        W += alpha * dp[1:]

    R, rho, X, pr = _eval_residual_reduced(mu, W, phi, S, delta, A, e, lm)
    rn_final = 0.0
    for k in range(n_p):
        rn_final += R[k] * R[k]
    rn_final = np.sqrt(rn_final)

    P = A.shape[0]
    hv = np.empty(Fd)
    for s in range(Fd):
        hv[s] = np.log(max(X[s], 1e-300)) - 0.5 * X[s] + 0.5
    A_mix = 0.0
    for i in range(P):
        A_mix += rho[i] * (np.log(max(rho[i], 1e-300)) - lm[i])
    A_lin = 0.0
    for i in range(P):
        A_lin += e[i] * rho[i]
    A_assoc = 0.0
    for s in range(Fd):
        A_assoc += hv[s] * pr[s]
    f = (A_mix + A_lin + A_assoc) / 7.0
    return f, mu, W, rn_final, it + 1


def solve_free_energy_curve_reduced(
    phi_grid,
    A_row,
    e_linear,
    log_mult,
    S,
    delta_small_np,
    *,
    mu_init=0.0,
    W_init=None,
    max_newton=50,
    tol=1e-10,
):
    """
    Compute f(phi) over a grid using the numba-accelerated reduced solver.

    Returns dict with keys: phi, f, mu_W (warm-start state for next call).
    """
    phi_grid = np.asarray(phi_grid, dtype=np.float64)
    A_row = np.ascontiguousarray(np.asarray(A_row, dtype=np.float64))
    e_linear = np.ascontiguousarray(np.asarray(e_linear, dtype=np.float64))
    log_mult = np.ascontiguousarray(np.asarray(log_mult, dtype=np.float64))
    S = np.ascontiguousarray(np.asarray(S, dtype=np.float64))
    delta_small_np = np.ascontiguousarray(np.asarray(delta_small_np, dtype=np.float64))

    P = len(A_row)
    Fd = S.shape[1]
    phi_lo, phi_hi = float(A_row.min()), float(A_row.max())

    fvals = np.full(len(phi_grid), np.nan)
    mus   = np.full(len(phi_grid), np.nan)
    mu = float(mu_init)
    W = np.zeros(Fd, dtype=np.float64) if W_init is None else W_init.copy()

    for idx, phi in enumerate(phi_grid):
        if phi < phi_lo - 1e-6 or phi > phi_hi + 1e-6:
            continue
        f, mu, W, rn, _ = _solve_single_phi_reduced(
            phi, mu, W, S, delta_small_np, A_row, e_linear, log_mult,
            max_newton=max_newton, tol=tol,
        )
        if rn < 1e-4:
            fvals[idx] = f
            mus[idx]   = mu

    return {"phi": phi_grid, "f": fvals, "mu": mus, "mu_W": (mu, W.copy())}




# ---------------------------------------------------------------------------
# Spinodal / binodal detection via mu(phi) — no smoothing required
# ---------------------------------------------------------------------------



