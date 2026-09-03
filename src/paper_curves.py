"""
Verbatim port of the curve-smoothing pipeline in ``notebooks/stick_l_critical_lines.ipynb``,
so that anything compared against the published figure is put through exactly the same
treatment.

The paper figure is built as follows.

MC line (cells 7 / 8)
    parse every ``theta* : (eps_nd, eps_sp, ...)`` record from
    ``results_logs/all_trace_coexist_{stick,l}.out``;
    x = eps_nd (= eps_c), y = eps_sp (= eps_a);
    drop non-finite and x < 0; for the L-shape additionally drop x < 0.56;
    then ``gaussian_smooth_xy_no_edge_curvature(x, y, sigma=50, n=...)``;
    for the L-shape, ``trim_at_floor(..., eps=0.02)``.

SAFT-P line (cells 4 / 6 feeding 7 / 8)
    load ``spinodal_{stick,l}_shaped_reduced_scan.npz``;
    per-x minima (min eps_a at each eps_c) restricted to eps_c <= 1.75;
    ``UnivariateSpline(k=3, s=0.5*len(xb), ext=3)`` sampled on 800 points;
    then the same sigma=50 Gaussian smoothing (and floor trim for the L).

Per-system parameters (they differ between the two cells):

    stick : n = 800, kfit = 12, no x filter,   no floor trim
    L     : n = 900, kfit = 10, keep x >= 0.56, floor trim at eps = 0.02

Note that ``sigma`` is in *grid points* of an n-point uniform grid spanning whatever x-range
that particular curve happens to have, so the physical smoothing width is not the same for
the MC curve and the SAFT-P curve.  That is reproduced here rather than corrected, because
the point is to compare against the published figure.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy.interpolate import UnivariateSpline
from scipy.ndimage import gaussian_filter1d

from mc_critical_lines import parse_trace

# per-system settings, taken from the two cells
PAPER = {
    "stick": dict(n=800, kfit=12, x_min=None, floor_trim=False,
                  trace="all_trace_coexist_stick.out",
                  npz="spinodal_stick_shaped_reduced_scan.npz"),
    "L":     dict(n=900, kfit=10, x_min=0.56, floor_trim=True,
                  trace="all_trace_coexist_l.out",
                  npz="spinodal_l_shaped_reduced_scan.npz"),
}
SIGMA = 50.0


def _fit_slope(x, y, k=10, side="left"):
    k = min(int(k), len(x))
    if k < 2:
        return 0.0
    xx, yy = (x[:k], y[:k]) if side == "left" else (x[-k:], y[-k:])
    A = np.vstack([xx, np.ones_like(xx)]).T
    m, b = np.linalg.lstsq(A, yy, rcond=None)[0]
    return float(m)


def gaussian_smooth_xy_no_edge_curvature(x, y, *, sigma=50.0, n=800, xlim=None,
                                         pad_mult=4, kfit=10):
    """Verbatim from the figure notebook."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    idx = np.argsort(x)
    x, y = x[idx], y[idx]

    if xlim is None:
        x0, x1 = float(x.min()), float(x.max())
    else:
        x0, x1 = map(float, xlim)
        keep = (x >= x0) & (x <= x1)
        x, y = x[keep], y[keep]

    xg = np.linspace(x0, x1, int(n))
    yg = np.interp(xg, x, y)

    pad = int(max(8, pad_mult * sigma))
    mL = _fit_slope(xg, yg, k=kfit, side="left")
    mR = _fit_slope(xg, yg, k=kfit, side="right")
    dx = xg[1] - xg[0]

    left_x = xg[0] - dx * np.arange(pad, 0, -1)
    right_x = xg[-1] + dx * np.arange(1, pad + 1)
    left_y = yg[0] + mL * (left_x - xg[0])
    right_y = yg[-1] + mR * (right_x - xg[-1])

    yg_pad = np.concatenate([left_y, yg, right_y])
    ygs_pad = gaussian_filter1d(yg_pad, sigma=float(sigma), mode="nearest")
    return xg, ygs_pad[pad:-pad]


def trim_at_floor(x, y, floor=0.0, eps=2e-2):
    """Verbatim from the figure notebook."""
    x = np.asarray(x)
    y = np.asarray(y)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    idx = np.where(y <= (floor + eps))[0]
    if idx.size == 0:
        return x, y
    k = int(idx[0])
    return x[:k + 1], y[:k + 1]


def paper_mc_curve(geometry: str, logs_dir: str) -> Tuple[np.ndarray, np.ndarray]:
    """The MC curve exactly as the paper figure draws it."""
    cfg = PAPER[geometry]
    ec, ea = parse_trace(f"{logs_dir}/{cfg['trace']}")
    keep = np.isfinite(ec) & np.isfinite(ea) & (ec >= 0)
    if cfg["x_min"] is not None:
        keep &= ec >= cfg["x_min"]
    ec, ea = ec[keep], ea[keep]
    xg, yg = gaussian_smooth_xy_no_edge_curvature(
        ec, ea, sigma=SIGMA, n=cfg["n"], kfit=cfg["kfit"])
    if cfg["floor_trim"]:
        xg, yg = trim_at_floor(xg, yg, floor=0.0, eps=0.02)
    return xg, yg


def lower_edge_from_npz(path: str, *, x_max: float = 1.75, decimals: int = 6):
    """Per-eps_c minimum eps_a from a published grid scan, as in cells 4 / 6."""
    pts = np.load(path)["points"]
    y, x = pts[:, 0], pts[:, 1]
    xq = np.round(x, decimals)
    xb, yb = [], []
    for xv in np.unique(xq):
        if xv > x_max:
            continue
        xb.append(float(xv))
        yb.append(float(y[xq == xv].min()))
    xb, yb = np.array(xb), np.array(yb)
    o = np.argsort(xb)
    return xb[o], yb[o]


def paper_saftp_curve(geometry: str, data_dir: str, *, npz: str | None = None,
                      s_per_point: float = 0.5, k: int = 3):
    """The published SAFT-P curve exactly as the paper figure draws it."""
    cfg = PAPER[geometry]
    path = f"{data_dir}/{npz or cfg['npz']}"
    xb, yb = lower_edge_from_npz(path)
    spl = UnivariateSpline(xb, yb, k=k, s=s_per_point * len(xb), ext=3)
    xgrid = np.linspace(xb.min(), xb.max(), 800)
    ygrid = spl(xgrid)
    xg, yg = gaussian_smooth_xy_no_edge_curvature(
        xgrid, ygrid, sigma=SIGMA, n=cfg["n"], kfit=cfg["kfit"])
    if cfg["floor_trim"]:
        xg, yg = trim_at_floor(xg, yg, floor=0.0, eps=0.02)
    return xg, yg


def apply_paper_smoothing(geometry: str, eps_c, eps_a):
    """
    Put a sparse, directly computed critical line through the same smoothing chain,
    so it is drawn on the same footing as the published curves.

    The spline step is skipped (it exists to de-noise a grid scan; a bisected line is
    already smooth) but the Gaussian pass and the floor trim are applied.
    """
    cfg = PAPER[geometry]
    eps_c = np.asarray(eps_c, float)
    eps_a = np.asarray(eps_a, float)
    m = np.isfinite(eps_c) & np.isfinite(eps_a)
    xg, yg = gaussian_smooth_xy_no_edge_curvature(
        eps_c[m], eps_a[m], sigma=SIGMA, n=cfg["n"], kfit=cfg["kfit"])
    if cfg["floor_trim"]:
        xg, yg = trim_at_floor(xg, yg, floor=0.0, eps=0.02)
    return xg, yg


def eval_curve(xg, yg, eps_c_query) -> np.ndarray:
    """Sample a smoothed curve at requested eps_c; NaN outside its support."""
    q = np.asarray(eps_c_query, float)
    out = np.interp(q, xg, yg, left=np.nan, right=np.nan)
    return out
