import numpy as np
import torch
from typing import Optional, Tuple, Dict, Any, List
from itertools import permutations, product
import torch.nn.functional as F

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
            "Eeff_sum": 0.0,
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
                acc["Eeff_sum"] *= fac
                acc["species_sum"] *= fac
                for k in list(acc["boundary_sum"].keys()):
                    acc["boundary_sum"][k] *= fac
            acc["E0"] = Eeff
            acc["best_cfg"] = cfg

        contrib = float(np.exp(-(Eeff - acc["E0"])))
        acc["w"] += contrib
        acc["Eeff_sum"] += contrib * Eeff
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
        intra_bonds.append(float(acc["Eeff_sum"] / w))

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
    J: Optional[np.ndarray] = None,
    mu: Optional[np.ndarray] = None,
    *,
    epsilon_patch: Optional[np.ndarray] = None,
    canonicalize_by_boundary_edges: bool = True,
    return_class_members: bool = False,
) -> Any:
    """
    Build 2x2x2 canonical cube macrostates from 6-patch species.

    Expected patch ordering per species row:
      [N, E, S, W, top(+z), bottom(-z)].

    Canonicalization:
      - boundary-edge canonicalization only
      - rotational degeneracies are removed by the 24 proper cube rotations

    Returns (same contract style as build_plaquettes_by_species):
      eps_small, intra_bonds, cube_to_species, m_patch,
      patch_to_species, patch_to_small, cube_configs, mult_arr
    """
    if not canonicalize_by_boundary_edges:
        raise ValueError("build_cubes_species supports only canonicalize_by_boundary_edges=True.")

    if J is None and epsilon_patch is None:
        raise ValueError("Provide either J or epsilon_patch.")
    if J is None:
        J = epsilon_patch
    elif epsilon_patch is not None:
        J_arr = np.asarray(J, dtype=np.float64)
        E_arr = np.asarray(epsilon_patch, dtype=np.float64)
        if J_arr.shape != E_arr.shape or not np.allclose(J_arr, E_arr):
            raise ValueError("If both J and epsilon_patch are provided, they must match.")

    patches = np.asarray(patches, dtype=np.int64)
    J = np.asarray(J, dtype=np.float64)

    if patches.ndim != 2 or patches.shape[1] != 6:
        raise ValueError(f"patches must have shape (n_species, 6), got {patches.shape}")

    n_species = int(patches.shape[0])
    n_patch_types = int(J.shape[0])
    if J.shape != (n_patch_types, n_patch_types):
        raise ValueError(f"J must be square, got {J.shape}")

    if patches.min() < 0 or patches.max() >= n_patch_types:
        raise ValueError("patch ids in patches must lie in [0, J.shape[0]).")

    mu_vec = np.zeros(n_species, dtype=np.float64) if mu is None else np.asarray(mu, dtype=np.float64)
    if mu_vec.shape != (n_species,):
        raise ValueError(f"mu must have shape ({n_species},), got {mu_vec.shape}")

    # Hard guard: exhaustive 8-corner enumeration scales as n_species**8.
    total_cfg = int(n_species ** 8)
    if total_cfg > 1_000_000_000:
        raise ValueError(
            f"Too many cube configurations ({total_cfg}). "
            "Reduce species count before exhaustive cube canonicalization."
        )

    # Patch index convention
    N, E, Sdir, W, TOP, BOT = 0, 1, 2, 3, 4, 5

    # Corner order (matches the prototype indexing):
    #   0:000, 1:010, 2:100, 3:110, 4:001, 5:011, 6:101, 7:111
    idx_to_coord = np.array(
        [
            [0, 0, 0],
            [0, 1, 0],
            [1, 0, 0],
            [1, 1, 0],
            [0, 0, 1],
            [0, 1, 1],
            [1, 0, 1],
            [1, 1, 1],
        ],
        dtype=np.int64,
    )
    coord_to_idx = {tuple(c.tolist()): i for i, c in enumerate(idx_to_coord)}

    # Direction vectors in the same axis convention as the corner coordinates:
    # axis-0: N/S, axis-1: W/E, axis-2: bottom/top
    dir_vecs = np.array(
        [
            [-1, 0, 0],  # N
            [0, 1, 0],   # E
            [1, 0, 0],   # S
            [0, -1, 0],  # W
            [0, 0, 1],   # top
            [0, 0, -1],  # bottom
        ],
        dtype=np.int64,
    )
    vec_to_dir = {tuple(v.tolist()): i for i, v in enumerate(dir_vecs)}

    def _encode4(a: int, b: int, c: int, d: int) -> int:
        n = n_patch_types
        return int(a) * (n ** 3) + int(b) * (n ** 2) + int(c) * n + int(d)

    def _face_ids_from_corner_patches(cp: np.ndarray) -> Tuple[int, int, int, int, int, int]:
        """
        cp: (8,6) patch ids for cube corners in idx_to_coord order.

        Face IDs are encoded exactly in the prototype ordering:
          (top, right, bottom, left, front, back).
        """
        p000, p010, p100, p110, p001, p011, p101, p111 = cp
        top = _encode4(p000[N], p010[N], p001[N], p011[N])
        right = _encode4(p000[W], p100[W], p001[W], p101[W])
        bottom = _encode4(p100[Sdir], p110[Sdir], p101[Sdir], p111[Sdir])
        left = _encode4(p110[E], p010[E], p111[E], p011[E])
        front = _encode4(p101[TOP], p111[TOP], p001[TOP], p011[TOP])
        back = _encode4(p000[BOT], p010[BOT], p100[BOT], p110[BOT])
        return (top, right, bottom, left, front, back)

    def _cfg_energy(cfg: Tuple[int, ...]) -> float:
        s000, s010, s100, s110, s001, s011, s101, s111 = cfg
        p000 = patches[s000]
        p010 = patches[s010]
        p100 = patches[s100]
        p110 = patches[s110]
        p001 = patches[s001]
        p011 = patches[s011]
        p101 = patches[s101]
        p111 = patches[s111]

        # 12 nearest-neighbor bonds inside a 2x2x2 cube
        bx1 = J[p000[E], p010[W]]
        bx2 = J[p100[E], p110[W]]
        bx3 = J[p001[E], p011[W]]
        bx4 = J[p101[E], p111[W]]

        by1 = J[p000[Sdir], p100[N]]
        by2 = J[p010[Sdir], p110[N]]
        by3 = J[p001[Sdir], p101[N]]
        by4 = J[p011[Sdir], p111[N]]

        bz1 = J[p000[TOP], p001[BOT]]
        bz2 = J[p010[TOP], p011[BOT]]
        bz3 = J[p100[TOP], p101[BOT]]
        bz4 = J[p110[TOP], p111[BOT]]

        return float(
            bx1 + bx2 + bx3 + bx4 +
            by1 + by2 + by3 + by4 +
            bz1 + bz2 + bz3 + bz4 +
            mu_vec[s000] + mu_vec[s010] + mu_vec[s100] + mu_vec[s110] +
            mu_vec[s001] + mu_vec[s011] + mu_vec[s101] + mu_vec[s111]
        )

    def _cfg_species_counts(cfg: Tuple[int, ...]) -> np.ndarray:
        return np.bincount(np.asarray(cfg, dtype=np.int64), minlength=n_species).astype(np.float64)

    # Precompute 24 proper cube rotations as corner/patch-direction permutations.
    rotations: List[Tuple[np.ndarray, np.ndarray]] = []
    seen_rot = set()
    for perm in permutations((0, 1, 2)):
        for signs in product((-1, 1), repeat=3):
            R = np.zeros((3, 3), dtype=np.int64)
            for row, col in enumerate(perm):
                R[row, col] = signs[row]
            if int(round(np.linalg.det(R))) != 1:
                continue

            key = tuple(R.ravel().tolist())
            if key in seen_rot:
                continue
            seen_rot.add(key)

            corner_perm = np.empty(8, dtype=np.int64)  # old corner -> new corner
            for old_i, c in enumerate(idx_to_coord):
                centered = 2 * c - 1
                centered_new = R @ centered
                new_c = ((centered_new + 1) // 2).astype(np.int64)
                corner_perm[old_i] = coord_to_idx[tuple(new_c.tolist())]

            dir_perm = np.empty(6, dtype=np.int64)  # old dir -> new dir
            for old_d, v in enumerate(dir_vecs):
                new_v = tuple((R @ v).tolist())
                dir_perm[old_d] = vec_to_dir[new_v]

            rotations.append((corner_perm, dir_perm))

    if len(rotations) != 24:
        raise RuntimeError(f"Expected 24 proper cube rotations, found {len(rotations)}.")

    def _canonical_boundary_key(cfg: Tuple[int, ...]) -> Tuple[int, int, int, int, int, int]:
        cp = patches[np.asarray(cfg, dtype=np.int64)]  # (8,6)
        keys = []
        for corner_perm, dir_perm in rotations:
            cp_rot = np.empty((8, 6), dtype=np.int64)
            cp_rot[corner_perm[:, None], dir_perm[None, :]] = cp
            keys.append(_face_ids_from_corner_patches(cp_rot))
        return min(keys)

    def _new_acc() -> Dict[str, Any]:
        return {
            "E0": np.inf,
            "w": 0.0,
            "Eeff_sum": 0.0,
            "species_sum": np.zeros(n_species, dtype=np.float64),
            "boundary_sum": {},
            "best_cfg": None,
        }

    def _acc_add(acc: Dict[str, Any], cfg: Tuple[int, ...]):
        Eeff = float(_cfg_energy(cfg))

        if Eeff < acc["E0"]:
            if np.isfinite(acc["E0"]):
                fac = float(np.exp(-(acc["E0"] - Eeff)))
                acc["w"] *= fac
                acc["Eeff_sum"] *= fac
                acc["species_sum"] *= fac
                for k in list(acc["boundary_sum"].keys()):
                    acc["boundary_sum"][k] *= fac
            acc["E0"] = Eeff
            acc["best_cfg"] = cfg

        contrib = float(np.exp(-(Eeff - acc["E0"])))
        acc["w"] += contrib
        acc["Eeff_sum"] += contrib * Eeff
        acc["species_sum"] += contrib * _cfg_species_counts(cfg)

        cp = patches[np.asarray(cfg, dtype=np.int64)]
        for fid in _face_ids_from_corner_patches(cp):
            acc["boundary_sum"][int(fid)] = acc["boundary_sum"].get(int(fid), 0.0) + contrib

    groups: Dict[Tuple[int, int, int, int, int, int], Dict[str, Any]] = {}
    members_map: Dict[Tuple[int, int, int, int, int, int], set] = {}
    for cfg in np.ndindex(*(n_species,) * 8):
        key = _canonical_boundary_key(cfg)
        acc = groups.get(key)
        if acc is None:
            acc = _new_acc()
            groups[key] = acc
            if return_class_members:
                members_map[key] = set()
        _acc_add(acc, cfg)
        if return_class_members:
            members_map[key].add(tuple(int(x) for x in cfg))

    keys_in_order: List[Tuple[int, int, int, int, int, int]] = []
    cube_configs = []
    intra_bonds = []
    cube_to_species_rows = []
    boundary_avg_list = []
    mult_arr = []

    for key, acc in groups.items():
        w = float(acc["w"])
        if (not np.isfinite(acc["E0"])) or (w <= 0.0):
            continue
        keys_in_order.append(key)
        cube_configs.append(acc["best_cfg"])
        intra_bonds.append(float(acc["Eeff_sum"] / w))
        cube_to_species_rows.append(acc["species_sum"] / w)
        boundary_avg_list.append({int(t): float(s / w) for t, s in acc["boundary_sum"].items()})
        # Degeneracy is included in Boltzmann aggregation; class linear term is the Boltzmann-averaged effective energy
        mult_arr.append(1.0)

    cube_configs = np.asarray(cube_configs, dtype=np.int64)
    intra_bonds = np.asarray(intra_bonds, dtype=np.float64)
    cube_to_species = np.asarray(cube_to_species_rows, dtype=np.float64)
    mult_arr = np.asarray(mult_arr, dtype=np.float64)

    P_macro = int(cube_configs.shape[0])
    all_face_ids = sorted({int(t) for bavg in boundary_avg_list for t in bavg.keys()})
    unique_face_ids = np.asarray(all_face_ids, dtype=np.int64)
    face_id_to_idx = {int(fid): idx for idx, fid in enumerate(unique_face_ids.tolist())}

    pts, pss, mps = [], [], []
    for i in range(P_macro):
        bavg = boundary_avg_list[i]
        for fid, cnt in bavg.items():
            pts.append(i)
            pss.append(face_id_to_idx[int(fid)])
            mps.append(float(cnt))

    patch_to_species = np.asarray(pts, dtype=np.int64)
    patch_to_small = np.asarray(pss, dtype=np.int64)
    m_patch = np.asarray(mps, dtype=np.float64)

    # Keep epsilon_small consistent with the provided prototype.
    n = n_patch_types
    ids = unique_face_ids
    p1 = ids // (n ** 3)
    rem = ids % (n ** 3)
    p2 = rem // (n ** 2)
    rem = rem % (n ** 2)
    p3 = rem // n
    p4 = rem % n

    eps_small = (
        J[p1[:, None], p2[None, :]] +
        J[p2[:, None], p1[None, :]] +
        J[p3[:, None], p4[None, :]] +
        J[p4[:, None], p3[None, :]]
    ).astype(np.float64, copy=False)

    if return_class_members:
        cube_class_members: List[Tuple[Tuple[int, ...], ...]] = []
        for key in keys_in_order:
            members = tuple(sorted(members_map.get(key, set())))
            cube_class_members.append(members)
        return (
            eps_small,
            intra_bonds,
            cube_to_species,
            m_patch,
            patch_to_species,
            patch_to_small,
            cube_configs,
            mult_arr,
            cube_class_members,
        )

    return (
        eps_small,
        intra_bonds,
        cube_to_species,
        m_patch,
        patch_to_species,
        patch_to_small,
        cube_configs,
        mult_arr,
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
    f = (A_lin + A_assoc + A_mix) / 4.0
    if return_X:
        return f, X
    return f
