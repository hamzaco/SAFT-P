"""
saftp_chirality.py -- SAFT-P plaquette free-energy / spinodal / binodal scan for the
two-isomer (geometric-isomer) system, packaged out of
notebooks/chirality_saftp_calculations.ipynb so it can be run headless on the cluster.

Only change relative to the notebook: the composition grid lower bound is now the
keyword `phi_lo` (was hard-coded 0.01).  The published Fig. 7 used phi_lo=0.01 with
n_phi=51, for which the binodal for T <= 0.8 landed exactly on the grid endpoints;
use a smaller phi_lo and a larger n_phi to converge those branches.

    find_spinodal_chirality_species(..., n_phi=201, phi_lo=0.002, ...)
"""
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.show = lambda *a, **k: plt.close("all")
from plaquette_by_species import *


class Particle:
    def __init__(self,index,patches=[],mu=0):
        self.patches=patches
        self.mu=mu
        self.index=index
        
    def get(self,i):
        return self.patches[i]
    def set_mu(self,mu):
        self.mu=mu
def get_species_list_ind(species):


    letter_to_index = {}  # mapping from letter to unique integer index
    next_index = 1
    particles = []
    letter_to_index['0']=0
    ind=0
    for s in species:
        n = len(s)
        # Create a base patch array of length 4 filled with zeros (0 means "no $
        base_array = np.zeros(4, dtype=int)
        # Right-align the string: fill positions 4 - n to 3 with the correspond$
        for i, letter in enumerate(s):
            if letter not in letter_to_index:
                letter_to_index[letter] = next_index
                next_index += 1
            base_array[4 - n + i] = letter_to_index[letter]
       
        
        # Generate all four rotations of the base array.
        unique_rotations = set()
        rotation_list = []  # to preserve an order (if needed)
        for r in range(4):
            rotated = np.roll(base_array, -r)  # rotate left by r positions
            tup = tuple(rotated.tolist())
            if tup not in unique_rotations:
                unique_rotations.add(tup)
                rotation_list.append(rotated)
        
        # For each distinct rotation, create a Particle object.
        for rot in rotation_list:
            particles.append(Particle(ind,patches=np.array(rot)))
            ind+=1
    
    return particles
# --------------------------------------------------------------------------
# Helper functions
# --------------------------------------------------------------------------

def detect_outliers_iqr(y, factor=2.0):
    """
    Detect outliers using IQR (Interquartile Range) method.
    Returns a boolean mask where True indicates outliers.
    """
    import numpy as np
    y_np = np.asarray(y)
    q1 = np.percentile(y_np, 25)
    q3 = np.percentile(y_np, 75)
    iqr = q3 - q1
    if iqr < 1e-10:
        # If IQR is too small, use MAD-based detection instead
        return detect_outliers_mad(y, factor=factor)
    lower_bound = q1 - factor * iqr
    upper_bound = q3 + factor * iqr
    outliers = (y_np < lower_bound) | (y_np > upper_bound)
    return outliers

def detect_outliers_mad(y, factor=3.0):
    """
    Detect outliers using Median Absolute Deviation (MAD) method.
    More robust than IQR for small datasets.
    Returns a boolean mask where True indicates outliers.
    """
    import numpy as np
    y_np = np.asarray(y)
    median = np.median(y_np)
    mad = np.median(np.abs(y_np - median))
    if mad < 1e-10:
        # No variation, no outliers
        return np.zeros_like(y_np, dtype=bool)
    threshold = factor * mad
    outliers = np.abs(y_np - median) > threshold
    return outliers

def detect_outliers_local(y, window=5, factor=2.0):
    """
    Detect outliers by comparing each point to its local neighbors.
    More robust for detecting isolated spikes in smooth data.
    Returns a boolean mask where True indicates outliers.
    """
    import numpy as np
    y_np = np.asarray(y)
    n = len(y_np)
    outliers = np.zeros(n, dtype=bool)
    
    for i in range(n):
        # Get local window around point i
        start = max(0, i - window // 2)
        end = min(n, i + window // 2 + 1)
        local_window = y_np[start:end]
        # Exclude the current point from local statistics
        local_window = np.concatenate([local_window[:i-start], local_window[i-start+1:]])
        
        if len(local_window) < 2:
            continue
        
        local_median = np.median(local_window)
        local_mad = np.median(np.abs(local_window - local_median))
        
        if local_mad > 1e-10:
            deviation = np.abs(y_np[i] - local_median) / local_mad
            outliers[i] = deviation > factor
    
    return outliers

def loess_derivs(x, y, *, bandwidth=0.15, order=2, ignore_outliers=True, outlier_method='local', outlier_factor=2.5, trim_edges=True, edge_trim_fraction=0.05):
    """
    Compute LOESS derivatives, optionally ignoring outliers and trimming edge artifacts.
    
    Parameters:
    - ignore_outliers: If True, exclude outliers from smoothing computation
    - outlier_method: 'iqr', 'mad', or 'local' (default: 'local' for better spike detection)
    - outlier_factor: Threshold factor for outlier detection (default: 2.5)
    - trim_edges: If True, trim problematic edge points and use extrapolation (default: True)
    - edge_trim_fraction: Fraction of points at each end to treat as edges (default: 0.05)
    """
    x = torch.as_tensor(x, dtype=torch.float64)
    y = torch.as_tensor(y, dtype=torch.float64)
    n = x.shape[0]
    
    # Detect outliers if requested
    outlier_mask = None
    if ignore_outliers:
        if outlier_method == 'iqr':
            outlier_mask = detect_outliers_iqr(y.numpy(), factor=outlier_factor)
        elif outlier_method == 'mad':
            outlier_mask = detect_outliers_mad(y.numpy(), factor=outlier_factor)
        elif outlier_method == 'local':
            outlier_mask = detect_outliers_local(y.numpy(), window=7, factor=outlier_factor)
        else:
            raise ValueError(f"Unknown outlier_method: {outlier_method}")
        outlier_mask = torch.as_tensor(outlier_mask, dtype=torch.bool)
    
    f0 = torch.zeros(n, dtype=torch.float64)
    f1 = torch.zeros(n, dtype=torch.float64)
    f2 = torch.zeros(n, dtype=torch.float64)
    
    # Identify edge regions
    edge_size = max(1, int(n * edge_trim_fraction))
    edge_mask = torch.zeros(n, dtype=torch.bool)
    if trim_edges and n > 2 * edge_size:
        edge_mask[:edge_size] = True
        edge_mask[-edge_size:] = True
    
    # Use adaptive bandwidth: larger at edges to smooth out boundary effects
    h_base = bandwidth * (x.max() - x.min())
    
    for i in range(n):
        # Increase bandwidth at edges for better smoothing
        if trim_edges and edge_mask[i]:
            h = h_base * 1.5  # 50% larger bandwidth at edges
        else:
            h = h_base
        
        d = (x - x[i]).abs()
        w = torch.clamp(1 - (d / h) ** 3, min=0) ** 3
        
        # Zero out weights for outliers if ignoring them
        if ignore_outliers and outlier_mask is not None:
            w = w * (~outlier_mask).float()
            # If current point is an outlier, use interpolated value from neighbors
            if outlier_mask[i]:
                # For outliers, use weighted average of non-outlier neighbors
                valid_mask = ~outlier_mask & (w > 1e-10)
                if valid_mask.sum() > 0:
                    # Interpolate from neighbors
                    w_valid = w[valid_mask]
                    if w_valid.sum() > 1e-10:
                        # Weighted interpolation
                        f0[i] = (y[valid_mask] * w_valid).sum() / w_valid.sum()
                    else:
                        # Fallback to nearest neighbor
                        nearest_idx = torch.argmin(d[valid_mask])
                        valid_indices = torch.where(valid_mask)[0]
                        f0[i] = y[valid_indices[nearest_idx]]
                    # Set derivatives to NaN for outliers
                    f1[i] = torch.nan
                    f2[i] = torch.nan
                    continue
        
        dx = x - x[i]
        cols = [torch.ones_like(dx)]
        if order >= 1:
            cols.append(dx)
        if order >= 2:
            cols.append(dx ** 2)
        X = torch.stack(cols, dim=1)
        XTW = (X.T * w)
        
        # Only use non-outlier points for fitting
        valid_weights = w > 1e-10
        if valid_weights.sum() < order + 1:
            # Not enough points, use interpolation
            f0[i] = y[i]
            f1[i] = torch.nan
            f2[i] = torch.nan
            continue
        
        try:
            beta = torch.linalg.solve(XTW @ X, XTW @ y)
            f0[i] = beta[0]
            f1[i] = beta[1] if order >= 1 else torch.nan
            f2[i] = (2 * beta[2]) if order >= 2 else torch.nan
        except:
            # If solve fails, use simple interpolation
            f0[i] = y[i]
            f1[i] = torch.nan
            f2[i] = torch.nan
    
    # Post-process edges: smooth transition and extrapolate using interior points
    if trim_edges and n > 2 * edge_size:
        # Left edge: extrapolate from interior points
        interior_start = edge_size
        interior_end = min(edge_size + 3, n)
        if interior_end > interior_start:
            # Fit linear extrapolation from interior
            x_left = x[interior_start:interior_end]
            y_left = f0[interior_start:interior_end]
            # Linear fit for extrapolation
            dx_int = (x_left[1:] - x_left[:-1]).mean() if len(x_left) > 1 else x[1] - x[0]
            if abs(dx_int) > 1e-10:
                # Use slope from first two interior points
                slope_left = (y_left[1] - y_left[0]) / (x_left[1] - x_left[0]) if len(y_left) > 1 else 0.0
                for i in range(edge_size):
                    if i < len(y_left):
                        # Extrapolate
                        f0[i] = y_left[0] + slope_left * (x[i] - x_left[0])
                        # Use interior derivative
                        if not torch.isnan(f1[interior_start]):
                            f1[i] = f1[interior_start]
                        if not torch.isnan(f2[interior_start]):
                            f2[i] = f2[interior_start]
        
        # Right edge: extrapolate from interior points
        interior_start = max(0, n - edge_size - 3)
        interior_end = n - edge_size
        if interior_end > interior_start:
            x_right = x[interior_start:interior_end]
            y_right = f0[interior_start:interior_end]
            if len(y_right) > 1:
                # Use slope from last two interior points
                slope_right = (y_right[-1] - y_right[-2]) / (x_right[-1] - x_right[-2])
                for i in range(n - edge_size, n):
                    idx = i - (n - edge_size)
                    # Extrapolate
                    f0[i] = y_right[-1] + slope_right * (x[i] - x_right[-1])
                    # Use interior derivative
                    if not torch.isnan(f1[interior_end - 1]):
                        f1[i] = f1[interior_end - 1]
                    if not torch.isnan(f2[interior_end - 1]):
                        f2[i] = f2[interior_end - 1]
    
    return f0, f1, f2


class StepTracker:
    def __init__(self):
        self.data = []
    def log(self, **kwargs):
        self.data.append(kwargs)
    def as_list(self):
        return self.data


# --------------------------------------------------------------------------
# Binodal extraction via convex hull
# --------------------------------------------------------------------------
from scipy.spatial import ConvexHull

def extract_binodals_from_convex_envelope(phi, f, *, hull_tol=0.0, coexist_tol=1e-5, min_gap_points=6):
    """
    Find binodal points from the convex envelope of f(phi).
    Returns segments where the common tangent lies below the curve.
    """
    phi = np.asarray(phi, float)
    f = np.asarray(f, float)
    pts = np.column_stack([phi, f])
    try:
        hull = ConvexHull(pts)
    except Exception:
        return dict(phi=phi, f=f, hull_idx=[], segments=[])
    
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
        below_min = float(np.min(diff[mask]))
        if barrier < coexist_tol:
            continue
        out.append(dict(phi1=float(phi1), phi2=float(phi2), mu=float(mu_line),
                        barrier=barrier, below_min=below_min, n_points=int(j - i)))
    return dict(phi=phi, f=f, hull_idx=hull_idx.tolist(), segments=out)


def best_binodal_segment(result, *, prefer="largest_barrier"):
    """Select the best binodal segment from convex hull analysis."""
    segs = result.get("segments", [])
    if not segs:
        return None
    if prefer == "widest":
        key = lambda d: (d["phi2"] - d["phi1"])
    elif prefer == "most_points":
        key = lambda d: d["n_points"]
    else:  # largest_barrier
        key = lambda d: ((d["phi2"] - d["phi1"]), d["barrier"])
    return max(segs, key=key)


def find_binodal_lower_envelope(phi, f, *, prefer="widest", min_gap_points=6, min_barrier=0.0):
    """
    Find a binodal candidate using the lower convex envelope of f(phi).

    Returns the best *skipping* segment of the lower hull. A skipping segment
    is a hull edge (phi1->phi2) that has at least one original sample point
    strictly between phi1 and phi2; these are the common-tangent coexistence
    candidates.

    Parameters
    - prefer: "widest" (default) or "largest_barrier"
      - "widest" picks the outermost coexistence region (usually what you want
        for a phase diagram)
      - "largest_barrier" can pick an inner/spurious tangent when the curve is
        only weakly non-convex or noisy
    - min_gap_points: require at least this many grid points between phi1, phi2
    - min_barrier: require barrier >= this value (helps ignore numerical noise)
    """
    phi = np.asarray(phi, float)
    f = np.asarray(f, float)
    n = len(phi)

    order = np.argsort(phi)
    phi_sorted = phi[order]
    f_sorted = f[order]

    # Build lower convex hull (monotone/Graham-scan style)
    lower = []
    for i in range(n):
        while len(lower) >= 2:
            p1 = lower[-2]
            p2 = lower[-1]
            p3 = (phi_sorted[i], f_sorted[i])
            cross = (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0])
            if cross <= 0:
                lower.pop()
            else:
                break
        lower.append((phi_sorted[i], f_sorted[i]))

    lower = np.asarray(lower, float)

    candidates = []
    for k in range(len(lower) - 1):
        phi1, f1 = lower[k]
        phi2, f2 = lower[k + 1]

        # count interior points on the original grid
        mask = (phi > phi1 + 1e-12) & (phi < phi2 - 1e-12)
        n_between = int(np.sum(mask))
        if n_between < min_gap_points:
            continue

        slope = (f2 - f1) / (phi2 - phi1)
        line_vals = f1 + slope * (phi[mask] - phi1)
        diff = f[mask] - line_vals
        barrier = float(np.max(diff))
        if barrier < min_barrier:
            continue

        candidates.append(dict(
            phi1=float(phi1), phi2=float(phi2),
            f1=float(f1), f2=float(f2),
            mu=float(slope), barrier=barrier,
            width=float(phi2 - phi1), n_between=n_between,
        ))

    if not candidates:
        return None

    if prefer == "largest_barrier":
        return max(candidates, key=lambda d: (d["barrier"], d["width"]))
    # default: outermost coexistence
    return max(candidates, key=lambda d: (d["width"], d["barrier"]))

def subtract_linear_part(phi, f, ignore_outliers=True, outlier_method='local', outlier_factor=2.5):
    """
    Subtract linear fit from f to reveal double-well structure.
    Optionally ignores outliers when fitting the linear part.
    
    Parameters:
    - ignore_outliers: If True, exclude outliers from linear fit
    - outlier_method: 'iqr', 'mad', or 'local' (default: 'local' for better spike detection)
    - outlier_factor: Threshold factor for outlier detection (default: 2.5)
    """
    phi = np.asarray(phi, float)
    f = np.asarray(f, float)
    
    # Detect outliers if requested
    outlier_mask = None
    if ignore_outliers:
        if outlier_method == 'iqr':
            outlier_mask = detect_outliers_iqr(f, factor=outlier_factor)
        elif outlier_method == 'mad':
            outlier_mask = detect_outliers_mad(f, factor=outlier_factor)
        elif outlier_method == 'local':
            outlier_mask = detect_outliers_local(f, window=7, factor=outlier_factor)
        else:
            raise ValueError(f"Unknown outlier_method: {outlier_method}")
        
        # Use non-outlier points for linear fit
        valid_mask = ~outlier_mask
        if valid_mask.sum() < 2:
            # Not enough points, use all points
            valid_mask = np.ones_like(phi, dtype=bool)
        phi_fit = phi[valid_mask]
        f_fit = f[valid_mask]
    else:
        phi_fit = phi
        f_fit = f
    
    # Linear fit: f = a*phi + b
    A = np.vstack([phi_fit, np.ones_like(phi_fit)]).T
    slope, intercept = np.linalg.lstsq(A, f_fit, rcond=None)[0]
    linear_part = slope * phi + intercept
    return f - linear_part, slope, intercept




def find_spinodal_chirality_species(
    patches, Ts, factor, *,mu=None,
    rho_empty=0.5, z=4, lr=0.05, tol=1e-4, rot90_species=None,use_4_patch=False,
    J_template=None, n_phi=101, phi_lo=0.01,
    canonicalize_by_boundary_edges=True, canonicalize_by_edges=True,undir=False,
    composition_key="components", species_to_component=None
):
    """
    Scan temperature Ts, find spinodals using SPECIES-BASED plaquettes.
    
    Plaquette identity = canonical species configuration (not edges).

    ``composition_key="components"`` (the default) puts the per-chemical-species
    composition into the class key, so a class can no longer mix ABEE-rich,
    BAEE-rich and solvent-rich microstates behind one boundary signature.  This
    matters more here than in the single-particle models: the constraint rows
    A1/A2/A3 below are exactly the per-component monomer fractions, and under
    the legacy key they were Boltzmann averages rather than exact counts.  The
    default component map is the orbit decomposition of ``rot90_species``, which
    for this model is precisely {ABEE rotations}, {BAEE rotations}, {vacancy}.
    Pass ``composition_key="none"`` to reproduce the published construction.
    
    Constraints:
      - A1 (ABEE rotations 0-3) = phi (variable)
      - A3 (CD rotations 8-11) = rho_cd (fixed)
      - A4 (vacancy = species 12) = rho_empty (fixed)
      - A2 (BAEE rotations 4-7) = 1 - phi - rho_cd - rho_empty (implied)
    """
    spinodal = [None] * len(Ts)
    bounds = [None] * len(Ts)
    binodals = [None] * len(Ts)
    
    n_patch_types = patches.max() + 1
    
    for j, T in enumerate(Ts):
        print(f"T = {T:.3f}")
        
        # Build J matrix for this temperature
        if J_template is not None:
            J = J_template.copy() / T
        else:
            J = np.zeros((n_patch_types, n_patch_types))
            J[1, 4] = -3; J[2, 4] = 0
            J[1, 5] = 0; J[2, 5] = -3
            J[3, 3] = -1
            J = (J + J.T) / T
        if mu is None:  
            mu = np.zeros(len(patches))
        
        # Build plaquettes with SPECIES-BASED grouping
        (eps_s, intra_bonds, plaq_to_species, m_patch, patch_to_species, patch_to_small,
         plaq_configs, mult) = build_plaquettes_by_species(
            patches, J, mu, rot90_species, use_4_patch=use_4_patch,undirected_edges=undir,
                canonicalize_by_boundary_edges=canonicalize_by_boundary_edges, 
                canonicalize_by_edges=canonicalize_by_edges,
                composition_key=composition_key,
                species_to_component=species_to_component
        )
        
        # Build A rows
        A_all = build_A_rows_from_plaq(plaq_to_species)
        n_species = A_all.shape[0]
        P_macro = A_all.shape[1]
        print(f"  n_plaq={P_macro}, n_species={n_species}")
        
        # Build composite A rows by summing over species groups
        # Divide by 4 to get monomer fractions (each plaquette has 4 monomers)
        A1 = A_all[0:4].sum(axis=0) / 4.0    # ABEE fraction
        A2 = A_all[4:8].sum(axis=0) / 4.0    # BAEE fraction
        A3 = A_all[8:9].sum(axis=0) / 4.0  # SSSS (vacancy) fraction
        
        A_rows = np.vstack([A1,A3])
        targets_template = np.array([np.nan, rho_empty], dtype=float)
        
        # Torch tensors
        patch_to_species_t = torch.from_numpy(patch_to_species).long()
        patch_to_small_t = torch.from_numpy(patch_to_small).long()
        eps_small_t = torch.tensor(-eps_s, dtype=torch.float64)
        m_patch_t = torch.from_numpy(m_patch).double()
        mult_t = torch.from_numpy(mult).double()
        e_linear_t = torch.tensor(intra_bonds, dtype=torch.float64)
        
        Δ_small = torch.expm1(eps_small_t).div(factor)
        def f_tilde_torch(rho_t):
            # NOTE: eps_small / intra_bonds are already scaled by 1/T upstream (J = J_template / T)
            # so residual_free_energy_by_species returns a dimensionless beta*F. Pass beta=1.
            f_val = residual_free_energy_by_species(
                rho_t, 1.0, z,
                patch_to_species_t, patch_to_small_t, eps_small_t, m_patch_t,
                e_linear_t, Δ_small, mult_t,
            )
            return f_val
        
        # Phi scan with fewer points for stability
        phis = np.linspace(phi_lo, 1.0-rho_empty-phi_lo, n_phi)
        
        # This is the expensive part (one constrained optimization per phi).
        curve = free_energy_curve_chirality(
            phis, A_rows, targets_template, variable_idx=0,
            f_tilde_torch=f_tilde_torch,
            device="cpu", dtype=torch.float64,
            steps=10000, lr=lr, tol_res=tol,
        )


        # ------------------------------------------------------------------
        # Diagnostic: do boundary patch-type marginals change with phi?
        # If they are (nearly) constant vs phi, then the association term (and
        # most energetic terms) are effectively phi-independent, so the model
        # cannot produce a widening coexistence region as T decreases.
        # ------------------------------------------------------------------
        try:
            with torch.no_grad():
                W_list = []
                for rho_np in curve["rhos"]:
                    rho_t = torch.tensor(rho_np, dtype=torch.float64)
                    W = torch.zeros(int(eps_small_t.shape[0]), dtype=torch.float64)
                    # expected boundary patch counts per patch type
                    W.index_add_(0, patch_to_small_t, m_patch_t * rho_t[patch_to_species_t])
                    W = W / W.sum().clamp_min(1e-30)
                    W_list.append(W.numpy())
                Wm = np.stack(W_list, axis=0)
                max_var = np.max(Wm, axis=0) - np.min(Wm, axis=0)
                print(f"  boundary patch frac max Δ over phi: {max_var.max():.3e}")
        except Exception as _e:
            print(f"  (diag skipped: {type(_e).__name__})")
        
        # ------------------------------
        # No ABEE<->BAEE symmetrization.
        # If interactions distinguish enantiomers, the free energy need not satisfy
        # f(phi) = f(phi_total - phi), and the coexistence curve can be tilted.
        # ------------------------------
        phi = np.asarray(curve['phi'], float)
        f_raw = np.asarray(curve['f'], float)
        phi_total = 1.0 - float(rho_empty)
        phi_star = 0.5 * phi_total
        #f_raw= f_raw[::-1]
        #phi=phi[::-1]
        # Subtract linear part for visualization
        f_detrend, slope, intercept = subtract_linear_part(phi, f_raw)

        # Find binodal using lower convex envelope (proper common tangent)
        # Use LOESS smoothed data to reduce noise effects
        f_smooth_raw, _, _ = loess_derivs(
            torch.tensor(phi), torch.tensor(f_raw), 
            bandwidth=0.15, trim_edges=True, edge_trim_fraction=0.05
        )
        f_smooth_raw = f_smooth_raw.numpy()

        # Find binodal from lower convex envelope.
        # Prefer the *widest* coexistence segment (outermost binodal) and
        # require a small minimum barrier to avoid noise-picked tangents.
        binodal_seg = find_binodal_lower_envelope(
            phi, f_smooth_raw,
            prefer="widest",
            min_gap_points=max(6, int(0.1 * len(phi))),
            min_barrier=1e-6,
        )
        
        # Use LOESS smoothing on detrended curve for visualization
        f_smooth_det, fp_smooth, fpp_smooth = loess_derivs(
            torch.tensor(phi), torch.tensor(f_detrend), 
            bandwidth=0.2, trim_edges=True, edge_trim_fraction=0.05
        )
        f_smooth_det = f_smooth_det.numpy()
        
        plt.figure(figsize=(12, 4))
        
        # Left: Detrended free energy (reveals double-well)
        plt.subplot(1, 3, 1)
        plt.plot(phi, f_detrend, 'o', ms=3, alpha=0.5, label='raw - linear')
        plt.plot(phi, f_smooth_det, '-', lw=2, label='LOESS smooth')
        if binodal_seg:
            plt.axvline(binodal_seg['phi1'], c='g', ls='--', lw=2, label='binodal')
            plt.axvline(binodal_seg['phi2'], c='g', ls='--', lw=2)
        plt.xlabel('phi (ABEE)'); plt.ylabel('f - linear')
        plt.legend()
        plt.title(f'T = {T:.3f} (detrended)')
        
        # Middle: Free energy (raw) with common tangent
        plt.subplot(1, 3, 2)
        plt.plot(phi, f_raw, 'o-', ms=3, alpha=0.35, label='raw f(phi)')
        plt.axvline(phi_star, c='k', ls=':', lw=1, label='phi* = (phi_total)/2')
        if binodal_seg:
            # Plot common tangent line (computed on LOESS-smoothed raw curve)
            phi1, phi2 = binodal_seg['phi1'], binodal_seg['phi2']
            f1 = np.interp(phi1, phi, f_smooth_raw)
            f2 = np.interp(phi2, phi, f_smooth_raw)
            plt.plot([phi1, phi2], [f1, f2], 'g-', lw=2, label='common tangent')
            plt.plot([phi1, phi2], [f1, f2], 'go', ms=8)
        plt.xlabel('phi (ABEE)'); plt.ylabel('f')
        plt.legend()
        plt.title('Free energy (raw)')
        
        # Right: Second derivative
        plt.subplot(1, 3, 3)
        plt.plot(curve['phi_inner'], -curve['fpp'].numpy(), label="-f''(phi)")
        plt.axhline(0, c='k', ls='--', alpha=0.5)
        plt.xlabel('phi'); plt.ylabel("-f''")
        plt.ylim([-5, 5])
        for sp in curve['spinodal_phis']:
            plt.axvline(sp, c='r', ls='--', alpha=0.5, 
                        label='spinodal' if sp == curve['spinodal_phis'][0] else '')
        if binodal_seg:
            plt.axvline(binodal_seg['phi1'], c='g', ls='-', lw=2, label='binodal')
            plt.axvline(binodal_seg['phi2'], c='g', ls='-', lw=2)
        plt.legend()
        plt.title('Curvature')
        
        plt.tight_layout()
        plt.show()
        
        # Store results
        phi_spi = np.asarray(curve["spinodal_phis"], dtype=float)
        if binodal_seg:
            print(
                f"  -> Binodal: phi1={binodal_seg['phi1']:.3f}, phi2={binodal_seg['phi2']:.3f}, "
                f"barrier={binodal_seg['barrier']:.2e}"
            )
            binodals[j] = binodal_seg
        if phi_spi.size > 1:
            print(f"  -> Spinodal at phi = {curve['spinodal_phis']}")
            spinodal[j] = curve
            bounds[j] = phi_spi
    
    return spinodal, bounds, binodals