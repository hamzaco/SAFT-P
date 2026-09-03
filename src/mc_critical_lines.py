"""
Parse the MC critical-line traces in ``results_logs/all_trace_coexist_*.out``.

Each converged continuation step logs

    theta* : (eps_nd, eps_sp, mu, Lambda, s, r)

and the critical line is the sequence of (eps_nd, eps_sp) pairs.  In the SAFT-P
notebooks the same two couplings are called (eps_c, eps_a):

    eps_nd  ==  eps_c   non-directional attraction between any two particles
    eps_sp  ==  eps_a   extra attraction between two sticky patches

which is the identification used in ``stick_l_critical_lines.ipynb`` (it plots
theta*[0] on x against theta*[1] on y, against the SAFT-P (eps_c, eps_a) scan).
No rescaling is involved.

The trace revisits eps_sp values during the continuation, so ``mc_line`` reduces
it to one point per eps_sp (median, plus a spread) for direct comparison with a
SAFT-P critical line evaluated the other way round.
"""

from __future__ import annotations

import re
from typing import Dict, Tuple

import numpy as np

_THETA = re.compile(r'^\s*(?:θ\*|theta\*)\s*:\s*\(([^)]*)\)')


def parse_trace(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Return (eps_c, eps_a) = (eps_nd, eps_sp) for every converged theta* record."""
    ec, ea = [], []
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            m = _THETA.match(line)
            if not m:
                continue
            parts = [p.strip() for p in m.group(1).split(",")]
            try:
                nd, sp = float(parts[0]), float(parts[1])
            except (ValueError, IndexError):
                continue
            if not (np.isfinite(nd) and np.isfinite(sp)) or nd < 0 or sp < 0:
                continue
            ec.append(nd)
            ea.append(sp)
    return np.array(ec), np.array(ea)


def mc_line(path: str, *, decimals: int = 3) -> Dict[str, np.ndarray]:
    """
    Collapse the trace to one critical point per eps_a (= eps_sp) value.

    Returns eps_a (sorted), eps_c (median over repeats), eps_c_lo / eps_c_hi
    (min / max over repeats at that eps_a) and n_rep.
    """
    ec, ea = parse_trace(path)
    key = np.round(ea, decimals)
    uniq = np.unique(key)
    med = np.empty(len(uniq))
    lo = np.empty(len(uniq))
    hi = np.empty(len(uniq))
    cnt = np.zeros(len(uniq), dtype=np.int64)
    for i, k in enumerate(uniq):
        v = ec[key == k]
        med[i] = np.median(v)
        lo[i] = v.min()
        hi[i] = v.max()
        cnt[i] = v.size
    return {"eps_a": uniq, "eps_c": med, "eps_c_lo": lo, "eps_c_hi": hi,
            "n_rep": cnt, "raw_eps_a": ea, "raw_eps_c": ec}




def mc_eps_a_at_eps_c(path: str, eps_c_query, **kw) -> np.ndarray:
    """
    Interpolate the MC critical eps_a at the requested eps_c values.

    The MC line is single-valued and monotonically decreasing in eps_c over the
    measured range, so it can be inverted by sorting on eps_c.
    """
    d = mc_line(path, **kw)
    order = np.argsort(d["eps_c"])
    return np.interp(np.asarray(eps_c_query, float),
                     d["eps_c"][order], d["eps_a"][order])


