# SAFT-P — cluster-resolved associating-fluid theory for patchy particles

Code and data for *Cluster-Resolved Associating-Fluid Theory for
Patch-Geometry-Dependent Phase Behavior in Patchy Particle Mixtures*.

Contains the SAFT-P plaquette free-energy theory (spinodal and binodal scans on
square, triangular and cubic lattices), the lattice Monte-Carlo simulations used
as benchmarks, and the notebooks that turn both into the published figures.

## Layout

```
src/                       Python modules and SLURM job scripts
notebooks/                 analysis and figure-generating notebooks
data/                      derived numerical results (.npz, .json, .csv)
figures/                   generated figures
results_logs/              raw simulation and scan logs
transfer_matrix_entropy/   exact strip transfer-matrix entropy study
mc_isomer/                 raw isomer Wang–Landau output
```

## Install

```bash
pip install -r requirements.txt
```

## Running

Notebooks put `../src` on `sys.path` in their first cell, read from `../data`
and `../results_logs`, and write to `../figures`. Launch Jupyter from the
repository root so the relative paths resolve:

```bash
jupyter lab
```

The cluster jobs are submitted from `src/`, which the scripts resolve through
`$SLURM_SUBMIT_DIR`:

```bash
cd src

# cubic lattice: build the cache, then scan
BASE_PATCH=1,1,1,0,0,0 sbatch build_cube_cache_streaming_directional.sh
sbatch run_cube_spinodal_scan_assoc_lowmem.sh

# square-lattice finite-size critical line
bash submit_mc_line.sh          # see MC_LINE_README.md

# isomer Monte-Carlo and binodal
sbatch submit_mc_isomer.sh
sbatch submit_saftp_binodal.sh
```

The cube cache is large — ~24.5 GB at 13 species. `CASE_TAG` defaults to the
base patch so two geometries do not collide; `EXPAND_WAVES=N` runs the expand
shards in sequential groups if memory is tight.

## Cluster classes

Every builder (`plaquette_by_species`, `plaquette_nxn_by_species`,
`plaquette_by_species_hexagon`, and the cube builder) groups microstates into
classes by their outgoing boundary signature, canonicalised over the point group
of the cluster. The signature alone does not fix the composition, because
interior sites are invisible from the boundary, so the builders take a
`composition_key`:

- `"components"` (default) — class key is (boundary signature, count per
  chemical component), which separates classes of different composition while
  leaving the rotational collapse untouched;
- `"species"` — one component per species; well defined only when the rotation
  does not relabel species;
- `"none"` — the boundary-only key used in the first submission, kept so those
  numbers can be regenerated.

`src/plaquette_composition.py` holds the shared component-map machinery and
`src/check_composition_key.py` is the verification suite.

## Further documentation

- `src/MC_LINE_README.md` — the finite-size critical-line campaign
- `src/MC_LINE_RESULTS.md` — its results
- `transfer_matrix_entropy/exact_strip_entropy_SI.md` — the strip entropy study

## License

MIT — see `LICENSE`.
