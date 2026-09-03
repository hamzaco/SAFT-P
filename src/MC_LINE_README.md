# Finite-size critical **line** — `mc_line_*`

The notebook sweep (`notebooks/mc_sims_l_shaped.ipynb`, `main(eps_ns, eps_sp_prev, mu)`)
ported to the cluster and parameterised by lattice size, so the whole critical
line ε_nd,c(ε_d) can be measured at **L = 10, 20, 30** and extrapolated to
L → ∞.

This is *not* the same thing as `mc_fss_*`:

| | `mc_fss_*` | `mc_line_*` (this) |
|---|---|---|
| question | how far does ε_nd,c move with L **at one ε_d**? | how far does the **whole line** move with L? |
| ε_d | fixed (1.96) | swept, the notebook's continuation |
| estimator | grid scan at fixed ε_d (L-BFGS-B does not move there) | the notebook's L-BFGS-B, which does move in continuation |
| anchor | logged L=10 point at that ε_d | exact lattice-gas point at ε_d=0, or logged L=10 per segment |

> **Note.** The `mc_fss_*` campaign is not part of this repository — the published
> finite-size numbers come from `mc_line_*` alone. References to `mc_fss_core.py` /
> `mc_fss_point.py` below are historical.

Both are wanted: `mc_fss_*` gives Δ_FS at the state point the paper argues
about; `mc_line_*` shows that Δ_FS does not blow up somewhere else along the
line.

---

## Files

| file | role |
|---|---|
| `mc_line_core.py` | pipeline: MC driving, reweighting, mixed-field fit, one segment of the continuation |
| `mc_line_sweep.py` | CLI — one (system, L, segment, seed) per invocation; also `--benchmark` |
| `mc_line_manifest.py` | cuts the sweep into segments and gives each an anchor from `results_logs/all_trace_coexist_*.out` |
| `run_mc_line_pilot.sh` | **run this first**; four checks, each of which can invalidate the campaign |
| `submit_mc_line.sh` | builds manifests, submits one array per L, prints the core-hour cost |
| `run_mc_line.sh` | the array task; checkpoints and chains itself across walltime |
| `mc_line_merge.py` | stitches segments → line per L → L→∞ extrapolation, CSV + figures |

`mc_line_core.py` imports only `simulate_functions_min`. The estimator
functions are carried over from the notebook verbatim, byte for byte identical
to the ones in `mc_fss_core.py`; nothing in the theory was touched.

---

## Step 0: relocate at fixed ε_d before sweeping anything

`--relocate-only` runs one pass at a single ε_d — MC at the anchor, equal-area
μ, μ\* by reweighting, mixed-field (s, r) fit, then reweight in (ε_nd, μ) at
**fixed** ε_d — and stops. This is the notebook's first loop iteration, and at
larger L it is the single most informative thing you can run:

- the shift it reports **is** Δ_FS(L) at that ε_d, from one task;
- it says whether the logged L=10 anchor is a critical point at this L at all;
- it costs one continuation step instead of a whole segment.

It always uses the grid scan, never L-BFGS-B: at zero displacement there is no
d(ε_d) to put the optimiser on the steep flank, so L-BFGS-B provably returns
its starting point on the piecewise-constant objective. The scan window is the
finite-size rounding width (`--relocate-halfwidth` 0.08 at L=10, scaled as
L_ref/L, because the ESS-usable range narrows as the log-weights carry C ~ L²).

```bash
for L in 10 20 30; do
  python mc_line_sweep.py --system l --L $L --seed 1 \
      --eps-d-start 1.8 --relocate-only \
      --eps-nd0 1.3530862528281224 --mu0 5.201030 \
      --replicas 8 --ising-cache $WD/ising_ref.npz \
      --out $WD/l_L${L}_reloc180_s1.npz
done
python mc_line_merge.py --workdir $WD --outdir ../figures --system l
```

**Read `curvature` and `at_bound`.** A relocation that ends `AT-BOUND` with
`curvature = nan` walked to the edge of its window without finding an interior
minimum — the anchor is too far off or there is not enough statistics.
Widening the window is the wrong fix; it just lets it walk further. Add
statistics, or start from a closer anchor. `mc_line_merge.py` flags these as
`no-minimum`.

`--relocate-first` does the same pass and then continues into the sweep, which
reproduces the notebook's loop exactly (its first iteration has
`eps_sp == eps_sp_prev`). Without either flag the grid starts one step in.

## The mixed-field parameter s — read before trusting any Delta_FS

The fit objective was profiled over s for every L after each relocation. The
plotting helper for that is not shipped (it produced no paper figure), but the
three things it showed are all load-bearing and are recorded here.

**r is inert.** In the notebook's `fit_s_r_from_reweighted`, r never enters
`build_x_pdf` — it appears only in the `r_penalty*r**2` term. So the optimum is
always r = 0, Lambda = 1 - s*r is always exactly 1 (which is what the logged
theta* records show), and the fit is a one-dimensional problem in s.

**rho and u are nearly collinear**, measured corr = -0.997 at L=10 and -0.999
at L=20 for the bent system at eps_d = 1.8. So Var(rho - s*u) has a minimum at
s* = Cov(rho,u)/Var(u) ~ -0.25, where sigma(M) collapses to 4-7% of
sigma(rho). That point is a singularity of the objective, and **beyond it M has
flipped sign relative to rho** — a sign-flipped order parameter scores well
against a symmetric reference, so the objective has a spurious minimum on the
far side. `s_admissible_range()` cuts the search at that degeneracy and keeps
only the branch containing s = 0.

**s is only weakly identifiable at these sizes.** The bowl around s = 0 is
shallow, and at L=10 it can vanish entirely — the objective then slides
monotonically to whatever bound it is given. The notebook's L-BFGS-B, with its
1e-8 finite-difference step on a KDE-of-a-comb objective, does not survive
that: observed returning s = +5.00000 (its own bound, objective 0.846) where
the grid minimum of the same objective was 0.207. `--s-estimator auto` (the
default) keeps the notebook's optimiser, confines it to the physical branch,
and when the objective has no interior minimum there says so and falls back
rather than reporting a converged-looking number.

**So fix s across L.** Delta_FS(L) is a difference between critical points, and
it only means anything if every L uses the same definition of the order
parameter. Pass `--s0` / `S0=` with the logged trace value at that eps_d
(`mc_line_manifest.py --list-anchors` prints it; it is -0.0098 at eps_d = 1.8
for the bent system). This is what `mc_fss_point.py --fix-s` already does, and
FINDINGS notes the located critical point is insensitive to s at that
magnitude.

## Running the sweep

```bash
cd src

# 0. sanity-check the anchors that will be used
python mc_line_manifest.py --system l --L 20 --list-anchors

# 1. pilot.  ~1-2 h.  READ THE OUTPUT.
sbatch run_mc_line_pilot.sh

# 2. cost, before committing.  Pass the rate the pilot measured.
RATE=6e6 bash submit_mc_line.sh --dry-run

# 3. submit
bash submit_mc_line.sh

# 4. collect
python mc_line_merge.py --workdir /pool/hamza/mc_line --outdir ../figures \
    --drop-nonergodic
```

Useful overrides: `SIZES`, `SEEDS`, `EPS_D_MIN`, `EPS_D_MAX`, `EPS_D_STEP`,
`SWEEPS_REF`, `SCOUT_FRAC`, `CPUS`, `PARTITION`, `SYSTEMS="l stick"`.

---

## Cost — read this before submitting

Chain length scales as

    steps(L) = sweeps_ref · (L/10)^2.17 · L²          sweeps_ref = 6e6

i.e. the notebook's 6×10⁸ attempts at L=10, times constant sweeps-per-site,
times critical slowing down at z = 2.17. Measured throughput is ~5–6×10⁶
attempts/s/core and is independent of L (the lattice fits in cache at every
size here), so:

| L | steps/replica | 1 chain | 1 ε_d step | per 0.1 in ε_d |
|---|---|---|---|---|
| 10 | 6.0e8 | 2 min | 3 min | 15 min |
| 20 | 1.1e10 | 39 min | 57 min | 4 h 45 |
| 30 | 5.9e10 | 3 h 21 | 4 h 57 | 24 h 45 |

("1 ε_d step" = ~6 short scout chains at `SCOUT_FRAC=0.08` + one full
production chain.) Replicas run in parallel, 8 per task, so those are wall
times per task.

**The full ε_d = 0 → 6 sweep at L = 10, 20, 30 with one seed costs ≈ 11,300
core-hours** — about 57 h of wall time if 200 cores are free continuously.
`submit_mc_line.sh --dry-run` prints this number for whatever scope you set.

Three ways to cut it, in order of how little they cost scientifically:

- `EPS_D_MAX=2.5` → **≈ 4,800 core-h**. Covers ε_d = 1.96, the point the
  fixed-ε_d campaign already reports, and the whole region the SAFT-P vs SAFT
  comparison actually turns on. This is the one I would take.
- `SIZES="10 20"` → drops L=30. Two sizes still fit
  ε_nd,c(L) = ε_nd,c(∞) + a·L^(−1/ν), but with zero degrees of freedom, so the
  fit residual stops being a check on the leading-order form and becomes
  identically zero. That check is most of the value of a third size.
- `EPS_D_STEP=0.05` → coarser continuation. Cheapest per unit of ε_d covered,
  and a larger displacement per step actually helps the L-BFGS-B estimator
  move (see below) — but it is a change to the published estimator and should
  be stated as one.

Seeds multiply the cost linearly, so the default is **one**. The cheaper noise
estimate is already built in: neighbouring segments overlap by `OVERLAP=0.04`,
so every overlap is an independent repeat at the same ε_d, and `mc_line_merge`
reports the spread within each grid bin as the error bar. Add seeds at L=10,
where they are nearly free, to confirm the two agree.

---

## How the parallelism works

Two levels:

1. **Replicas within a task.** 8 MC chains in a `fork` pool, one per core.
   Each worker reseeds numba's RNG through an `@njit` call — numba's
   `np.random` state is process-global *and inherited across `fork()`*, and
   `np.random.seed()` from Python does not touch it, so without this every
   replica would reproduce the identical chain while the error bars looked
   perfectly healthy. The pilot checks this explicitly.

2. **Segments across the array.** The continuation is sequential by
   construction — step i+1 starts from the state located at step i — so it is
   parallelised by *restarting*, exactly as the notebook cells
   `main(1.1039…, 4.18, 6.6654)` do. `mc_line_manifest.py` cuts
   [ε_d,min, ε_d,max] into segments and anchors each on the logged L=10
   criticality record at its left edge.

Segment width is scaled inversely with cost so tasks stay a few hours to a day:
0.40 at L=10, 0.20 at L=20, 0.10 at L=30.

---

## Caveats to state in the paper

**Anchoring.** Only the first segment, which starts at ε_d = 0, is anchor-free
at every L: there the model reduces to the plain 2D lattice gas and the
critical point is exactly ε_nd,c = 2 ln(1+√2) = 1.762747 with μ = 2ε_nd,c + ln g
(g = 2 collinear, 4 bent — the notebook used ln 4 for both, which is right only
for the bent case). Every other segment inherits the **L=10** anchor at its
left edge, so it reports how the line moves away from the L=10 line over its
own width rather than an absolute point walked in from ε_d = 0. The
segment-to-segment overlap is the check that this is harmless; if overlaps
disagree, the error bars in `_lines.csv` will show it.

**L-BFGS-B stalls.** `optimize_eps_mu_Mc_fixed_epsA_fixed_s` minimises over a
digitised histogram, so the objective is piecewise constant and its
finite-difference gradient is noise. In continuation this is usually harmless
— each step starts displaced by d(ε_d) on the steep flank — but not always:
the logged bent trace has runs of 3–4 consecutive steps with ε_nd frozen to 5
decimals. Every step now records `stalled`, and `--scan-fallback` (on by
default in `run_mc_line.sh`) redoes those steps with a grid scan of the *same*
objective, keeping the result only if its JS is lower. `mc_line_merge.py`
flags any step that stalled without being rescued.

**Ergodicity.** Each step records the number of dilute↔dense round trips per
replica. `min_crossings = 0` means that point is not ergodic on the chain
length used and is not a critical point, whatever number came out.
`mc_line_merge.py --drop-nonergodic` excludes those, and the pilot's
"steps with a non-tunnelling replica" line is the early warning. If L=30 shows
them, raise `SWEEPS_REF` rather than believing the numbers.

**ESS.** `ESS/n` below ~0.1 on a step means the reweighting is carried by a
handful of configurations. Plotted per step in `_diag.png`.

---

## Output

Per segment: `<system>_L<L>_seg<NNN>_s<seed>.npz` (all per-step columns +
diagnostics + config) and a matching `.jsonl` written step by step, so a
running job can be inspected without waiting for it.

After `mc_line_merge.py`:

- `mc_line_<system>_points.csv` — every step, every diagnostic, flagged
- `mc_line_<system>_lines.csv` — stitched ε_nd,c(ε_d) per L on a common grid
- `mc_line_<system>_extrap.csv` — ε_nd,c(∞), the slope a, and the RMS fit residual
- `mc_line_<system>_lines.png`, `_shift.png`, `_diag.png`
