# Finite-size scaling of the critical line — results

Campaign `mc_line_*`, workdir `/pool/hamza/mc_line_v6`, code version 6, seed 1.
Nine state points × three lattice sizes (L = 10, 20, 30), eight cores per task,
6×10⁶ sweeps at L = 10 scaled as (L/10)^2.17.

Figures: `figures/mc_line_v6_scaling.png`, `figures/mc_line_v6_line.png`.
Regenerate with

```bash
python mc_line_figure.py --workdir /pool/hamza/mc_line_v6 \
    --data-dir ../data --out ../figures/mc_line_v6      # add --with-theory for SAFT
```

---

## 1. What was measured

At each state point the critical coupling ε_nd,c was located **independently at
each lattice size**, at fixed ε_d, by the manuscript's own estimator: equal-area
μ, μ\* by reweighting, mixed-field fit of M = ρ − s·u against the 2D-Ising
universal order-parameter distribution P\*(x), then a scan in (ε_nd, μ)
minimising the Jensen–Shannon divergence to P\*(x). The field-mixing parameter s
was pinned per state point at the local median of the logged L = 10 trace, so
every size uses the same definition of the order parameter.

Nothing about the theory enters. These are simulation numbers.

## 2. Located critical points

| system | ε_d | L = 10 | L = 20 | L = 30 |
|---|---|---|---|---|
| bent | 1.02 | 1.522852 | 1.498500 | 1.494573 |
| bent | 1.80 | 1.337112 | 1.312351 | 1.306934 |
| bent | 2.60 | 1.243212 | 1.215601 | 1.210461 |
| collinear | 1.00 | 1.549185 | 1.520430 | 1.516278 |
| collinear | 1.80 | 1.287134 | 1.264390 | 1.261463 |
| collinear | 1.96 | 1.220545 | 1.196816 | 1.190892 |

All six are monotone in L, all carry an interior JS minimum with finite
curvature, ESS/n ≈ 1.0, and 1000–2000 dilute↔dense round trips per replica. No
diagnostic flags.

Two further points were run and are **excluded**: collinear ε_d = 2.58 and 3.00.
Both give a decay amplitude of the opposite sign and three times the magnitude
of every other point, while bent ε_d = 2.60 — essentially the same ε_d —
behaves normally. Their replica crossings *fall* with L (1300 → 444 and
1155 → 231) where every retained point's *rise*, and they sit in the stretch
above ε_d ≈ 2.5 where the logged collinear trace's own fit quality collapses
(median JS 0.20 against 0.01 below 2.5). They are ergodicity-limited, not
measurements. Bent ε_d = 1.96 is also excluded: the scan ran away from a
marginal anchor (logged JS 0.10) and hit the ±0.10 step clamp at both sizes,
reporting anchor − 0.100000 exactly. It has been replaced by ε_d = 1.84, which
is an exact logged point in a clean stretch.

## 3. The scaling — left figure

Three sizes and three unknowns solve

    ε_nd,c(L) = ε_nd,c(∞) + a · L^(−x)

exactly, so **x is determined by the data**, not assumed. The diagnostic is the
ratio (y₁₀ − y₂₀)/(y₂₀ − y₃₀), which equals exactly **3.00** when x = 1.

| system | ε_d | ratio | x | ε_nd,c(∞) | Δ_FS(10) | Δ_FS(30) |
|---|---|---|---|---|---|---|
| bent | 1.02 | 6.20 | 2.23 | 1.49190 | 0.03095 | 0.00267 |
| bent | 1.80 | 4.57 | 1.72 | 1.30157 | 0.03555 | 0.00537 |
| bent | 2.60 | 5.37 | 1.99 | 1.20632 | 0.03689 | 0.00414 |
| collinear | 1.00 | 6.93 | 2.41 | 1.51377 | 0.03541 | 0.00250 |
| collinear | 1.80 | 7.77 | 2.60 | 1.25990 | 0.02724 | 0.00157 |
| collinear | 1.96 | 4.01 | 1.50 | 1.18380 | 0.03675 | 0.00710 |

**x = 2.08 ± 0.38** over six points. Every measured ratio lies between 4.0 and
7.8; none is consistent with 3.00. Two geometries, ε_d spanning a factor 2.6,
six independent simulations — they agree.

The left panel of the scaling figure shows this directly: the six shift curves
on log–log axes, with L^(−1) and L^(−2) reference slopes. Every curve is
steeper than L^(−1) and clusters about L^(−2). The right panel plots the
per-point exponent against ε_d, with the mean and its spread shaded, and the
x = 1 line for reference.

### Why x ≈ 2 rather than 1

For a generic observable the finite-size shift of a critical coupling goes as
L^(−1/ν), which is L^(−1) in the 2D-Ising class. That is not what this
estimator measures. The Bruce–Wilding criterion locates the critical point by
matching a *scale-invariant* distribution — the standardised order-parameter
histogram — so the leading L^(−1/ν) term can cancel and the next correction is
what survives. A steeper decay is therefore what this class of estimator should
be expected to give, and L^(−1) is the wrong default here.

This is a statement about the estimator, not about the universality class. It
does not bear on ν.

## 4. Consequences

**The 1/L extrapolation over-shoots.** Forcing x = 1 places ε_nd,c(∞) too low
by 0.009–0.016, which is 19–33% of the finite-size shift itself:

| | Δ_FS(L = 10) | residual at L = 30 |
|---|---|---|
| exponent measured (x ≈ 2.1) | **+0.0338 ± 0.0035** | **+0.0039** |
| forcing x = 1 | +0.0466 ± 0.0039 | +0.0167 |

**L = 30 is essentially converged.** The residual finite-size error at L = 30 is
0.0039 with the measured exponent, against 0.0167 if you assume 1/L — a factor
of four. In the right-hand figure the L = 30 curve and the L → ∞ curve are
nearly indistinguishable; the shaded band is the difference between the two
extrapolation choices, and it is the dominant remaining uncertainty in
ε_nd,c(∞).

**The shift is a rigid translation.** Δ_FS(L = 10) = +0.0338 ± 0.0035 — a 10%
spread across both geometries and ε_d from 1.0 to 2.6. Normalising by ε_nd,c
makes the spread *worse* (13% versus 10%), so what is constant is the absolute
displacement in ε_nd, not a relative one. The critical line at L = 10 sits
above the thermodynamic-limit line by a roughly fixed amount, rather than being
distorted in shape. This is visible in the right-hand figure as four
near-parallel curves.

## 5. Two further scalings

Both are properties of the estimator rather than of the model, but both say the
critical point is progressively better *determined*, not merely better *placed*:

- **Agreement with the universal form improves as L^(−1.30 ± 0.22).** Mean JS at
  the located point runs 0.0099 at L = 10 to 0.0025 at L = 30 — 4.1× better.
- **The JS minimum sharpens as L^(−0.70 ± 0.27)**, measuring width as
  1/√curvature: 0.19–0.32 at L = 10 down to 0.09–0.13 at L = 30.

## 6. Caveats

**Zero degrees of freedom per point.** Three sizes and three unknowns means x is
measured but not *tested* at any single state point. What makes the result
credible is six independent points landing in the same place, and all six
rejecting x = 1 by the ratio test. A fourth size would convert each point into a
genuine test and shrink the ±0.38.

**L = 16 is the cheapest next run** — roughly 11 minutes per replica at L = 10
throughput, against 5.5 hours for L = 30. Adding it to the six retained points
is a few hundred core-hours and is the single highest-value measurement left.

**s is pinned, not fitted.** The mixed-field parameter is only weakly
identifiable at these sizes; it is held at the logged L = 10 local median so
that every lattice size uses the same order parameter. Letting it float per size
would make Δ_FS a comparison between different observables. The stored s-profile
diagnostics show the objective is nearly flat in s over the physical branch, so
the choice matters much less than fixing it consistently does.

**Above ε_d ≈ 2.5 the collinear system is not accessible at these chain
lengths.** Crossings fall with L there. Either raise the sweep budget for those
points or report the range as ε_d ≤ 2.6.

## 7. Provenance

Every result carries `code_version = 6` in its `.npz`. Version 6 fixed the
defect that produced the previous round of numbers: the relocation scan
profiled μ freely and let JS choose it, which left the located point off
coexistence. Because the standardised variable has zero mean by construction,
p_L·|c_L| = p_R·|c_R| identically, so a basin-mass imbalance alone pushes the
lighter peak outward and manufactures an apparent asymmetry. Measured at
ε_d = 1.8: masses 0.5027/0.4973 at L = 10 but 0.5252/0.4748 at L = 20, which
accounted for the entire 0.096 dense-peak displacement. Version 6 pins μ to the
equal-area value at every ε_nd in the scan, by bisection on the same importance
weights — no additional MC. After the fix, monotonicity went from 2/6 to 8/8 and
JS-improves-with-L from 0/6 to 8/8.

(The trace-quality audit and the per-point overlay plots used to check this were
one-off diagnostics and are not part of the retained pipeline.)
