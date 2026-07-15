# Chiral patchy-particle phase behavior — code & data

Theory (SAFT-γ / plaquette free energy, spinodal & binodal scans) and Monte-Carlo
simulations for stick-, L-, Mercedes/hexagon-, and cube-shaped patchy particles,
plus the figure-generating notebooks for the manuscript.

## Repository layout

```
.
├── src/            Python modules and HPC (SLURM) job scripts
├── notebooks/      Jupyter notebooks (analysis + figure generation)
├── data/           Input/derived numerical results (.npz, .json, .csv)
├── figures/        Generated figures (.png, .pdf)
├── results_logs/   Raw simulation/scan run logs (.out, MC trace .txt)
├── .gitignore
└── README.md
```

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


## Notes
- Dependencies: `numpy`, `scipy`, `matplotlib`, `numba`, `pandas`, and `torch`
  (used by some spinodal/Mercedes scans).
