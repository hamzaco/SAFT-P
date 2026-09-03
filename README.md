# Chiral patchy-particle phase behavior — code & data

Theory (SAFT-γ / plaquette free energy, spinodal & binodal scans) and Monte-Carlo
simulations for stick-, L-, Mercedes/hexagon-, and cube-shaped patchy particles,
plus the figure-generating notebooks for the manuscript.

## Repository layout

```
.
├── src/                             Python modules and HPC (SLURM) job scripts
├── notebooks/                       Jupyter notebooks (analysis + figure generation)
├── data/                            Input/derived numerical results (.npz, .json, .csv)
├── figures/                         The 20 figure files used in the paper
├── results_logs/                    Raw simulation/scan run logs (.out, MC trace .txt)
├── transfer_matrix_entropy/         Exact strip transfer-matrix study — SI Figs. S1, S2
├── mc_isomer/                       Raw isomer Wang–Landau MC output
├── LICENSE
├── requirements.txt
├── .gitignore
└── README.md
```

This repository holds the code, data and figures behind the published results.

### `src/`
Importable modules (`plaquette_by_species`, `plaquette_by_species_hexagon`,
`simulate_functions_min`, `saft_analysis`, `simulate_*_parallel`, …) and the
cube spinodal-scan pipeline (`cube_cache_builder_streaming_directional.py`,
`cube_spinodal_scan_representative_assoc_memory*.py`) with its two SLURM
launchers (`build_cube_cache_streaming_directional.sh`,
`run_cube_spinodal_scan_assoc_lowmem.sh`).

### `notebooks/`
Each notebook begins with a small setup cell that puts `../src` on
`sys.path`, so the module imports work from inside `notebooks/`. Notebooks read
inputs from `../data` and `../results_logs` and write figures to `../figures`.

## Running

Launch Jupyter from the repository root (or from `notebooks/`) so the relative
paths resolve:

```bash
jupyter lab            # then open notebooks/<name>.ipynb
```

The cube SLURM scripts expect to be submitted from `src/` (they resolve paths
from `$SLURM_SUBMIT_DIR`):

```bash
cd src
sbatch build_cube_cache_streaming_directional.sh
sbatch run_cube_spinodal_scan_assoc_lowmem.sh
```

## Composition-aware plaquette classes

Every cluster builder here (`plaquette_by_species`, `plaquette_nxn_by_species`,
`plaquette_by_species_hexagon`, and the cube builder) groups microstates into
classes by their outgoing boundary signature, canonicalised over the point group
of the cluster. Collapsing rotations is exact — a rotation is a symmetry and does
not change what a cluster is made of. The boundary signature on its own, however,
does not determine the **composition**: a site shows only its outward patches and
interior sites are invisible, so microstates with different particle counts used
to be merged into one class and Boltzmann-averaged into a fractional composition.

The builders now take `composition_key`:

- `"components"` (default) — class key = (boundary signature, count per chemical
  component). A component is a set of species closed under the rotation map; the
  default map is that map's orbit decomposition, so rotational isomers of one
  particle merge while chemically distinct species and the vacancy stay separate.
  Component counts are invariant under the point group, so the orientational
  collapse is untouched and only composition is separated.
- `"species"` — one component per species; only well defined when the rotation
  does not relabel species (the cube).
- `"none"` — the published boundary-only key, kept so the published numbers can
  be regenerated and regressed against.

`src/plaquette_composition.py` holds the shared component-map machinery, and
`src/check_composition_key.py` is the verification suite (legacy parity, integer
compositions, the non-interacting limit, rotation invariance):

```bash
cd src
mkdir -p /tmp/ref
git show HEAD:'src/plaquette_by_species.py'         > /tmp/ref/plaquette_by_species.py
git show HEAD:'src/plaquette_by_species_hexagon.py' > /tmp/ref/plaquette_by_species_hexagon.py
python check_composition_key.py /tmp/ref     # omit the argument to skip legacy parity
```

Effect on the systems in the manuscript:

| system | classes, `"none"` -> `"components"` | note |
|---|---|---|
| stick / L, 2x2 | 24 -> 24, 165 -> 165 | unchanged; every site is on the perimeter, so the published numbers stand |
| stick / L, 3x3 | 1,665 -> 3,330, 12,720 -> 25,440 | matches `interior_key="occupancy"`; reachable phi goes from [0.074, 1] / [0.089, 1] to [0, 1] |
| 7-site hexagon | 130 -> 260 | matches `split_center_vacancy=True`; reachable phi goes from [0.095, 1] to [0, 1] |
| chirality (ABEF/BAEF/solvent) | 1,044 -> 1,566 | the constraint rows A1/A2/A3 become exact monomer counts instead of half-integer averages |
| cube 2x2x2 | unchanged in practice | see below |

### The cube scans are unaffected

The cluster cube pipeline (`cube_cache_builder_streaming_directional.py` +
`cube_spinodal_scan_representative_assoc_memory_nu_fixed.py`, launched by
`build_cube_cache_streaming_directional.sh` /
`run_cube_spinodal_scan_assoc_lowmem.sh`) is self-contained: it imports only
numpy/scipy and never calls `build_cubes_species`, so `composition_key` does not
reach it and the SLURM scripts need no change.

Nor does it need one. Its boundary key already fixes the composition, because the
vacancy carries a dedicated patch type (`VACANCY_TYPE=2`) while every particle
rotation uses only patches 0/1. Each corner of a 2x2x2 cube exposes three faces,
so the boundary always reveals whether that corner is occupied — checked for the
production base patch `1,1,1,0,0,0` (13 species) and for `1,0,0,0,0,0`,
`1,1,0,0,0,0` and `1,1,1,1,0,0`: no exposed corner triple is ambiguous about
occupancy, and the built cache has `max|n_occ - round(n_occ)| = 0`. The residual
fractional part in `cube_to_species` is the split among rotational isomers of one
chemical species, which is exactly the orientational degeneracy that should be
averaged.

This argument depends on there being a single chemical species plus a vacancy. A
cube case with two chemically distinct particles (the analogue of the chirality
model) would need the composition key inside
`cube_cache_builder_streaming_directional.py`, a bump of
`_cube_cache_fingerprint(..., version=...)` so stale caches are rejected, and a
`--composition-key` passthrough in the launcher. `build_cubes_species` in
`plaquette_by_species.py` has the fix already, but nothing in this repository
calls it.

## Cube cache: memory at 13 species

`build_cube_cache_streaming_directional.sh` stores one row per cube
configuration, and the row count is `n_species**8`. Base patches `1,1,0,0,0,0`
and `1,1,1,0,0,0` both give 12 distinct rotations plus a vacancy = 13 species, so
that is **815,730,721 rows** -- against 65,536 at 4 species and 5.7M at 7. Those
builds were OOM-killed in the expand stage.

Fixed, with the cache contents unchanged (verified bit-for-bit against the
previous code through all four stages):

- `expand_group_shard` stored configurations as `int32`; species indices need
  only `uint8` at 13 species, and it built a list of per-class blocks and then
  `np.concatenate`d them, holding two full copies of the shard. It now sizes each
  class up front (`_cfg_count_from_key`, no configurations materialised) and
  fills one preallocated buffer. Peak for ten tasks on one node: 52 GB -> 6.5 GB.
- `merge_expand_shards` was where the 13-species builds actually died. It
  materialised `bond_a` and `bond_b` in full (19.6 GB) and histogrammed them
  afterwards inside `save_cube_cache`, which added an int64 row index and column
  index spanning the whole array; it also cast each loaded shard to int64 (a
  5.2 GB temporary for values that fit in a byte). `bond_hist` is now accumulated
  in the chunk loop, the raw pairs are never built, and shards are read in their
  stored dtype. Caches written this way carry `bond_hist` and no
  `bond_a`/`bond_b` -- the scanner never needed them and `cubes_from_cache` reads
  either form -- which also saves 19.6 GB of disk.
- `cfg`, `species_counts` and `bond_hist` are the three arrays that span every
  configuration, and they *are* the cache on disk. They are now allocated with
  `np.lib.format.open_memmap` directly in the destination directory and filled in
  place rather than assembled in RAM and written out afterwards, so merge-cache
  peak RAM drops from ~57 GB to a couple of GB. Measured at 7 species: peak RSS
  0.647 -> 0.263 GB with no wall-clock cost (16.5 s -> 15.9 s).
- That trades RAM for disk, and a memmap write to a full filesystem dies with a
  bare `Bus error` partway through, leaving a corrupt cache. merge-cache now
  checks free space up front and refuses with a clear message instead.
- The launcher had **no `#SBATCH --mem` line at all**, so it took the partition
  default. It now requests `--mem=0` (all memory on the node), prints a preflight
  size estimate before spending days on a build that cannot fit, and accepts
  `EXPAND_WAVES` to run the expand shards in sequential groups. More shards alone
  never helped: every expand task holds its whole shard at once, so N tasks on one
  node hold the entire configuration set no matter how finely it is sharded.

```bash
BASE_PATCH=1,1,1,0,0,0 sbatch build_cube_cache_streaming_directional.sh
BASE_PATCH=1,1,0,0,0,0 sbatch build_cube_cache_streaming_directional.sh
EXPAND_WAVES=4 BASE_PATCH=... sbatch build_cube_cache_streaming_directional.sh  # if expand is still tight
```

### One case per CASE_TAG

`CASE_TAG` used to default to the fixed string `cube_case` whatever `BASE_PATCH`
was, and every path derives from it -- `PATCHES_NPY`, `WORKDIR`, `KEYS_DIR`,
`EXPAND_DIR`, `MERGED_KEYS`, `FINAL_CACHE`. Two runs with different base patches
therefore shared all of them. Running them at the same time is destructive:
`CLEAN_BUILD=1` makes whichever job starts second `rm -rf` the first one's keys,
expand shards and final cache *while it is still using them*. And because the
patches file was reused when present rather than regenerated, the second job
silently built the **first** job's geometry under its own name.

`CASE_TAG` now defaults to the base patch (`cube_111000`, `cube_110000`), so the
two cases no longer collide by default. Because a shared `CASE_TAG` can still be
passed explicitly, the launcher also:

- **takes an exclusive lock** on `WORKDIR/.owner_job` before the destructive
  `CLEAN_BUILD` step, and refuses to start if another job holding it is still in
  the queue (a stale lock from a killed job is taken over automatically);
- **regenerates the patches file every run** into a job-unique temp name
  (`<case>_patches.new.$SLURM_JOB_ID.npy`) and installs it only after checking it
  against any existing file. A shared temp name was itself a race.

The check reports three distinct outcomes rather than one catch-all, because they
need different fixes: patches match; the existing file holds a *different
geometry* (wrong `CASE_TAG`); or the existing file is *unreadable* -- truncated by
a concurrent job or left over from an interrupted run, which surfaces as
`ValueError: This file contains pickled (object) data` and just needs deleting.

The final cache is ~24.5 GB on disk at 13 species. The preflight prints how much
is free under `FINAL_CACHE` and flags it if that is not enough.

## Notes
- Dependencies: see `requirements.txt`. **`torch` is required, not optional** —
  `plaquette_by_species`, `plaquette_by_species_hexagon`, `saftp_chirality` and
  `orig_solver_ref` import it at module level, and the constrained free-energy solver
  is written in torch tensors. Six modules will not import without it.
  `numba` is likewise required by `simulate_functions_min`.
- `cube_spinodal_scan_representative_assoc_memory_nu_fixed.py` opens with a
  `try`/`except ImportError` chain over four cache-builder module names; only
  `cube_cache_builder_streaming_directional` is present, and it is the one that resolves.
  The other three names are historical fallbacks and are expected to fail.
- `Archive.zip` and machine-generated caches (`__pycache__/`,
  `.ipynb_checkpoints/`, Numba `*.nbc`/`*.nbi`) are git-ignored.


## Which file produced which figure

| Paper figure | Produced by |
|---|---|
| Fig. 1 (coarse-graining schematic) | drawn externally |
| Fig. 2 (interaction schematic) | drawn externally |
| Fig. 3 stick + L critical lines | `notebooks/stick_l_critical_lines.ipynb` → `figures/phase_diagram_prl_stick.png`, `figures/phases_diagram_l_saftp_simulations.png` |
| Fig. 4 occupied fraction vs μ | `notebooks/phase_diagram_overlay_error_analysis_manual_points.ipynb` |
| Fig. 5 plaquette instability modes | `notebooks/plaquette_instability_local_motifs_figures.ipynb` → `figures/fig_motifs_{stick,lshape}_high_eps_a_phi050_mode{1,2}.png` |
| Fig. 6 isomer components | drawn externally |
| Fig. 7 isomer binodal | `notebooks/chirality_saftp_calculations.ipynb` (theory) + `notebooks/chirality_mc_simulations.ipynb` (MC) → `figures/binodal_excess_prl.png` |
| Fig. 8 triangular critical line | `notebooks/spinodal_hex_mercedes.ipynb`, run with the centre-aware class key — set `COMPOSITION_KEY = "components"` in the setup cell (`"none"` reproduces the earlier boundary-only key). The figure cell writes `figures/phase_diagram_mercedes.{png,pdf}`; that PNG is the published Fig. 8 and ships under the name `phase_diagram_mercedes_composition_resolved.png`. |
| Fig. 9 cubic critical lines | `src/cube_verification_figure.py` + `notebooks/threed_plots.ipynb`, on caches from `build_cube_cache_streaming_directional.sh` / `run_cube_spinodal_scan_assoc_lowmem.sh` |
| SI Fig. S1 strip entropy | `transfer_matrix_entropy/exact_strip_entropy.py` |
| SI Fig. S2 isolated-Z closure | `transfer_matrix_entropy/critical_point_comparison.py` |
| SI Fig. S3 ground state / structure factor | `notebooks/ground_state_l_particle.ipynb` |
| SI Fig. S4 2×2 vs 3×3 | `notebooks/critical_line_2x2_3x3_MC.ipynb`, `notebooks/spinodal_3x3_cluster_size_check.ipynb` → `figures/figSI_cluster_size_2x2_3x3_MC.png` |
| SI Figs. S5–S6 + Table S1 | the `src/mc_line_*` campaign (`submit_mc_line.sh` → `mc_line_manifest` → `mc_line_sweep` → `mc_line_merge` → `mc_line_figure`); numbers recorded in `src/MC_LINE_RESULTS.md` |
