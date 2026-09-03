#!/usr/bin/env python
"""
mc_line_sweep.py -- run one segment of the critical line eps_nd,c(eps_d) at
one lattice size.

One invocation == one SLURM array task == one (system, L, segment, seed).
Results go to a single .npz that mc_line_merge.py collects, plus a .jsonl trace
that is appended step by step so a running job can be inspected.

Examples
--------
# 3-minute plumbing check (tiny chains, 2 steps)
python mc_line_sweep.py --system l --L 10 --eps-d-start 0.0 --eps-d-end 0.04 \
    --sweeps-ref 2e4 --replicas 2 --samples-per-replica 200 --out /tmp/smoke.npz

# throughput benchmark and wall-time projection, no analysis
python mc_line_sweep.py --benchmark --L 10,20,30 --system l

# production segment, anchored on the logged L=10 trace
python mc_line_sweep.py --system l --L 20 --seed 1 \
    --eps-d-start 1.80 --eps-d-end 2.00 --eps-nd0 1.35309 --mu0 5.18903 \
    --replicas 8 --checkpoint $WORKDIR/ck_l_L20_seg09_s1.json \
    --out $WORKDIR/l_L20_seg09_s1.npz

# segment starting at eps_d = 0: omit --eps-nd0/--mu0 and the exact
# lattice-gas anchor 2 ln(1+sqrt 2), mu = 2 eps_nd + ln g is used
python mc_line_sweep.py --system l --L 10 --eps-d-start 0 --eps-d-end 0.4 \
    --out $WORKDIR/l_L10_seg00_s1.npz
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mc_line_core import (  # noqa: E402
    MC_LINE_VERSION,
    SYSTEMS, SweepConfig, run_segment, ising_reference, numba_seed,
    get_species_list_ind, make_lattices, run_ref_once_mu, lattice_gas_anchor,
)


def _hms(sec):
    sec = float(sec)
    d = int(sec // 86400)
    h = int((sec % 86400) // 3600)
    m = int((sec % 3600) // 60)
    s = int(sec % 60)
    return (f"{d}-{h:02d}:{m:02d}:{s:02d}" if d else f"{h:d}:{m:02d}:{s:02d}")


def _jsonable(o):
    if isinstance(o, dict):
        return {k: _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return o


def benchmark(system, Ls, target_seconds=6.0, verbose=True):
    """Measure MC throughput (single-site attempts/second) at each L.

    Run this before committing walltimes.  Throughput was ~6e6 attempts/s/core
    and essentially independent of L for the fixed-eps_d campaign (the lattice
    fits in cache at every size up to 32), but that is a property of the node,
    not a law -- measure it on the partition you are about to use.
    """
    sysdef = SYSTEMS[system]
    species = get_species_list_ind(sysdef["species"])
    n_species = len(species)
    species_mu_indices = np.arange(n_species - 1, dtype=np.int64)
    numba_seed(12345)
    rng = np.random.default_rng(0)
    e_nd, mu = lattice_gas_anchor(system)

    rows = []
    for L in Ls:
        lat = make_lattices(species, L, "middle", rng)[0]
        run_ref_once_mu(lat, species, species_mu_indices, e_nd, 1.0, mu,
                        steps=20_000, snapshot_interval=5_000,
                        buffer_size=64, block_size=1000,
                        empty_index=n_species - 1)          # warm the JIT
        steps = 2_000_000
        dt = 0.0
        for _ in range(6):
            t0 = time.perf_counter()
            run_ref_once_mu(lat, species, species_mu_indices, e_nd, 1.0, mu,
                            steps=steps,
                            snapshot_interval=max(steps // 100, 1),
                            buffer_size=128,
                            block_size=max(1000, steps // 1000),
                            empty_index=n_species - 1)
            dt = time.perf_counter() - t0
            if dt >= target_seconds * 0.5:
                break
            steps = int(steps * max(2.0, target_seconds / max(dt, 1e-3)))
        rate = steps / dt
        rows.append(dict(L=L, steps=steps, seconds=dt, steps_per_sec=rate))
        if verbose:
            print(f"[bench] L={L:3d}  {rate:.3e} attempts/s  "
                  f"({rate / L ** 2:.3e} sweeps/s)", flush=True)
    return rows


def project(rows, cfg: SweepConfig, n_scout_calls=6.0):
    """Wall time per eps_d step and per segment, given measured throughput.

    Budget per continuation step:
        n_scout_calls short chains (scout_frac of full length) + 1 full chain.
    6 is what the equal-area scout costs once mu is warm-started from the
    previous step; the first step of a segment costs more, and a noisy scout
    (short chains, small L) can cost twice that.  The number is printed so you
    can reconcile it against the real per-step wall time from the pilot.
    """
    equiv = n_scout_calls * cfg.scout_frac + 1.0
    n_steps = max(int(round((cfg.eps_sp_end - cfg.eps_sp_start)
                            / cfg.eps_sp_step)), 1)
    print("\n[project] wall time per replica (= per core; replicas run in "
          "parallel)")
    print(f"[project] steps(L) = {cfg.sweeps_ref:.3g} * (L/{cfg.L_ref})^"
          f"{cfg.z_exponent} * L^2;  {equiv:.2f} full-chain equivalents per "
          f"eps_d step")
    print(f"{'L':>4} {'steps/replica':>15} {'sweeps':>11} {'1 chain':>11} "
          f"{'1 eps_d step':>13} {'per 0.1 in eps_d':>18}")
    for r in rows:
        L = r["L"]
        steps = cfg.steps_for(L)
        t1 = steps / r["steps_per_sec"]
        tstep = t1 * equiv
        t01 = tstep * (0.1 / cfg.eps_sp_step)
        print(f"{L:>4} {steps:>15.4e} {steps / L ** 2:>11.3e} "
              f"{_hms(t1):>11} {_hms(tstep):>13} {_hms(t01):>18}")
    print(f"\n[project] this segment is {n_steps} steps of "
          f"{cfg.eps_sp_step:g} in eps_d")


def main(argv=None):
    p = argparse.ArgumentParser(
        description="One segment of the finite-size critical line")
    p.add_argument("--system", default="l", choices=sorted(SYSTEMS),
                   help="'l' = bent (mc_sims_l_shaped.ipynb), "
                        "'stick' = collinear")
    p.add_argument("--L", default="10",
                   help="lattice side; comma list allowed in --benchmark mode")
    p.add_argument("--seed", type=int, default=0)

    # --- segment ---------------------------------------------------------
    p.add_argument("--eps-d-start", type=float, default=0.0,
                   help="eps_d at the segment anchor (eps_sp in the notebook)")
    p.add_argument("--eps-d-end", type=float, default=0.4)
    p.add_argument("--eps-d-step", type=float, default=0.02,
                   help="notebook value 0.02")
    p.add_argument("--eps-nd0", type=float, default=None,
                   help="eps_nd at the anchor; omit at eps_d=0 to use the "
                        "exact lattice-gas value")
    p.add_argument("--mu0", type=float, default=None)
    p.add_argument("--s0", type=float, default=None,
                   help="hold the field-mixing parameter fixed at this value "
                        "instead of refitting each step")

    # --- statistics ------------------------------------------------------
    p.add_argument("--sweeps-ref", type=float, default=6.0e6,
                   help="sweeps at L=L_ref; 6e6 reproduces the notebook's "
                        "600e6 attempts at L=10")
    p.add_argument("--L-ref", type=int, default=10)
    p.add_argument("--z", type=float, default=2.17,
                   help="dynamic exponent used to scale sweeps with L")
    p.add_argument("--replicas", type=int, default=8)
    p.add_argument("--workers", type=int, default=0,
                   help="worker processes (0 -> --replicas)")
    p.add_argument("--samples-per-replica", type=int, default=6000)
    p.add_argument("--burn-frac", type=float, default=0.10)
    p.add_argument("--scout-frac", type=float, default=0.08,
                   help="chain length of the equal-area mu scout, as a "
                        "fraction of the production chain. 1.0 reproduces "
                        "the notebook exactly and costs ~4x more")
    p.add_argument("--scout-burn-frac", type=float, default=0.40)

    # --- estimator -------------------------------------------------------
    p.add_argument("--scout-step0", type=float, default=0.004)
    p.add_argument("--scout-tol", type=float, default=1e-1)
    p.add_argument("--ess-frac", type=float, default=0.40)
    p.add_argument("--sigma-x", type=float, default=0.03)
    p.add_argument("--r-penalty", type=float, default=1e-1)
    p.add_argument("--w-asym", type=float, default=2.0)
    p.add_argument("--opt-halfwidth-per-step", type=float, default=1.0,
                   help="(eps_nd, mu) optimiser box, in units of the eps_d "
                        "step. 1.0 with step 0.02 == the notebook's +-0.02")
    p.add_argument("--scan-fallback", action="store_true",
                   help="redo any step where L-BFGS-B did not move eps_nd "
                        "with a grid scan of the same objective")
    p.add_argument("--relocate-first", action="store_true",
                   help="open with a zero-displacement pass at eps_d_start: "
                        "MC at the anchor, then reweight in (mu, eps_nd) at "
                        "FIXED eps_d to re-locate the critical point at this "
                        "L.  This is the notebook's first loop iteration, and "
                        "it always uses the grid scan (L-BFGS-B cannot move "
                        "at zero displacement)")
    p.add_argument("--relocate-only", action="store_true",
                   help="do only that pass and stop -- run this at each L "
                        "before committing to a sweep")
    p.add_argument("--relocate-halfwidth", type=float, default=0.15,
                   help="eps_nd scan half-width. Scan wide; the ESS floor "
                        "decides which part of the window is usable")
    p.add_argument("--relocate-scale-with-L", action="store_true",
                   help="shrink the window as L_ref/L. Off by default: the "
                        "displacement being measured does not shrink that "
                        "fast, and at L=30 the scaled window did not reach "
                        "the critical point at all")
    p.add_argument("--scan-mu-mode", default="equal-area",
                   choices=["equal-area", "free"],
                   help="how mu is set at each eps_nd in the relocation scan. "
                        "'equal-area' (default) balances the two basins by "
                        "reweighting; 'free' lets JS choose it, which leaves "
                        "the located point off coexistence")
    p.add_argument("--relocate-ess-floor", type=float, default=0.15,
                   help="a grid point may only be chosen as the minimum if "
                        "ESS/n is at least this")
    p.add_argument("--relocate-mu-halfwidth", type=float, default=0.05)
    p.add_argument("--s-fallback", type=float, default=0.0,
                   help="s to use when the fit objective has no interior "
                        "minimum on the physical branch")
    p.add_argument("--s-estimator", default="auto",
                   choices=["lbfgs", "auto", "grid"],
                   help="how the mixed-field s is fitted. 'lbfgs' is the "
                        "notebook's call verbatim; 'auto' (default) keeps it "
                        "but falls back to a grid minimum of the same "
                        "objective when it terminates on its own bound, "
                        "which is what it does at L=10; 'grid' always uses "
                        "the grid. Ignored when --s0 is given")
    p.add_argument("--max-abs-eps-step", type=float, default=0.10)

    p.add_argument("--init", default="middle", choices=["mixed", "middle"],
                   help="'middle' reproduces the notebook initialisation")
    p.add_argument("--ising-cache", default=None)
    p.add_argument("--checkpoint", default=None,
                   help="json state file; a segment that runs out of walltime "
                        "resumes from here on resubmission")
    p.add_argument("--archive-raw", action="store_true",
                   help="store every raw sample of the last step")
    p.add_argument("--out", default=None)
    p.add_argument("--jsonl", default=None,
                   help="append one json record per eps_d step (default: "
                        "<out>.jsonl)")
    p.add_argument("--benchmark", action="store_true",
                   help="measure throughput, project wall times, then exit")
    p.add_argument("--verbose-inner", action="store_true",
                   help="print every mu scout / reweighting iteration")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args(argv)

    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
              "NUMEXPR_NUM_THREADS", "NUMBA_NUM_THREADS"):
        os.environ.setdefault(v, "1")

    verbose = not args.quiet
    Ls = [int(x) for x in str(args.L).split(",")]

    # --relocate-only == relocate pass with an empty continuation grid
    eps_d_end = args.eps_d_start if args.relocate_only else args.eps_d_end
    relocate_first = args.relocate_first or args.relocate_only

    cfg = SweepConfig(
        system=args.system, L=Ls[0], seed=args.seed,
        eps_sp_start=args.eps_d_start, eps_sp_end=eps_d_end,
        eps_sp_step=args.eps_d_step,
        relocate_first=relocate_first,
        relocate_halfwidth=args.relocate_halfwidth,
        relocate_mu_halfwidth=args.relocate_mu_halfwidth,
        relocate_scale_with_L=args.relocate_scale_with_L,
        relocate_ess_floor=args.relocate_ess_floor,
        scan_mu_mode=args.scan_mu_mode,
        s_estimator=args.s_estimator,
        s_fallback=args.s_fallback,
        eps_ns0=args.eps_nd0, mu0=args.mu0, s0=args.s0,
        sweeps_ref=args.sweeps_ref, L_ref=args.L_ref, z_exponent=args.z,
        n_replicas=args.replicas, n_workers=args.workers,
        samples_per_replica=args.samples_per_replica,
        burn_frac=args.burn_frac, scout_frac=args.scout_frac,
        scout_burn_frac=args.scout_burn_frac,
        scout_step0=args.scout_step0, scout_tol=args.scout_tol,
        ess_frac=args.ess_frac, sigma_x=args.sigma_x,
        r_penalty=args.r_penalty, w_asym=args.w_asym,
        opt_halfwidth_per_step=args.opt_halfwidth_per_step,
        scan_fallback=args.scan_fallback,
        max_abs_eps_step=args.max_abs_eps_step,
        init_mode=args.init, ising_cache=args.ising_cache,
        checkpoint=args.checkpoint, archive_raw=args.archive_raw,
        verbose_inner=args.verbose_inner,
    )

    if args.benchmark:
        rows = benchmark(args.system, Ls, verbose=verbose)
        project(rows, cfg)
        if args.out:
            np.savez_compressed(args.out, bench=json.dumps(_jsonable(rows)))
        return 0

    t0 = time.perf_counter()
    res = run_segment(cfg, verbose=verbose)
    res["wall_seconds"] = time.perf_counter() - t0

    pts = res["points"]
    print("\n" + "=" * 74)
    print(f"SEGMENT  system={res['system']}  L={res['L']}  seed={res['seed']}"
          f"  eps_d {res['eps_sp_start']:.4f} -> {res['eps_sp_end']:.4f}")
    if pts:
        print(f"  steps completed : {len(pts)}  "
              f"(complete={res['complete']})")
        rel = [r for r in pts if r["mode"] == "relocate"]
        if rel:
            r0 = rel[0]
            print(f"  RELOCATION at fixed eps_d={r0['eps_d']:.4f}, L={res['L']}")
            print(f"    anchor          : eps_nd={res['eps_nd_anchor']:.6f}"
                  f"  mu={res['mu_anchor']:.6f}")
            print(f"    relocated to    : eps_nd={r0['eps_nd']:.6f}"
                  f"  mu={r0['mu']:.6f}")
            print(f"    shift           : {r0['eps_nd'] - res['eps_nd_anchor']:+.6f}"
                  f"   <- this IS Delta_FS(L) at this eps_d")
            print(f"    JS at minimum   : {r0['js']:.5f}   "
                  f"curvature {r0['curvature']:.4g}")
            print(f"    field mixing s  : {r0['s']:.5f}")
        print(f"  last point      : eps_d={pts[-1]['eps_d']:.4f}  "
              f"eps_nd={pts[-1]['eps_nd']:.6f}  mu={pts[-1]['mu']:.6f}")
        n_stall = sum(1 for r in pts if r["stalled"])
        n_resc = sum(1 for r in pts if r["rescued_by_scan"])
        n_bound = sum(1 for r in pts if r["at_bound"])
        n_notun = sum(1 for r in pts if r["min_crossings"] < 1)
        print(f"  stalled steps   : {n_stall}/{len(pts)}"
              f"  (rescued by scan: {n_resc})")
        print(f"  at optimiser bound: {n_bound}/{len(pts)}")
        print(f"  steps with a non-tunnelling replica: {n_notun}/{len(pts)}"
              f"   <- 0 is what you want; anything else is not ergodic")
        cr = [r["min_crossings"] for r in pts]
        es = [r["ess_frac"] for r in pts]
        print(f"  min crossings   : {min(cr)} .. {max(cr)}")
        print(f"  ESS/n           : {min(es):.2f} .. {max(es):.2f}")
    else:
        print("  no steps completed")
    print(f"  wall            : {_hms(res['wall_seconds'])}")
    print("=" * 74, flush=True)

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        cols = ["eps_d", "eps_nd", "mu", "s", "r", "js", "ess_frac",
                "mean_rho", "min_crossings", "total_crossings", "mean_accept",
                "rho_spread", "Mc", "lam", "mu_star_ref", "match_L2",
                "asym_L2", "wall_seconds", "d_eps_d", "curvature", "s_fit"]
        payload = dict(
            code_version=MC_LINE_VERSION,
            s_fixed_at=(float(args.s0) if args.s0 is not None
                        else float("nan")),
            system=res["system"], L=res["L"], seed=res["seed"],
            eps_d_start=res["eps_sp_start"], eps_d_end=res["eps_sp_end"],
            eps_d_step=res["eps_sp_step"],
            eps_nd_anchor=res["eps_nd_anchor"], mu_anchor=res["mu_anchor"],
            complete=res["complete"], n_points=len(pts),
            steps_per_replica=res["steps_per_replica"],
            sweeps_per_replica=res["sweeps_per_replica"],
            n_replicas=res["n_replicas"],
            wall_seconds=res["wall_seconds"],
            points_json=json.dumps(_jsonable(pts)),
            config_json=json.dumps(_jsonable(vars(args))),
            centers_ref=res["centers_ref"], edges_ref=res["edges_ref"],
            P_ref=res["P_ref"],
        )
        for c in cols:
            payload[c] = np.array([float(r.get(c, np.nan)) for r in pts],
                                  float) if pts else np.zeros(0)
        for c in ("stalled", "at_bound", "rescued_by_scan", "opt_success",
                  "fit_success", "mu_scout_ok", "s_on_bound", "s_rescued",
                  "s_fixed", "s_flat", "s_unidentified", "eps_clamped"):
            payload[c] = np.array([bool(r.get(c, False)) for r in pts],
                                  bool) if pts else np.zeros(0, bool)
        for k, v in res.get("raw", {}).items():
            payload["raw_" + k] = np.asarray(v, dtype=np.float32)
        for k, v in res.get("diag_arrays", {}).items():
            payload["diag_" + k] = np.asarray(v)
        np.savez_compressed(args.out, **payload)
        print(f"[out] {args.out}", flush=True)

        jl = args.jsonl or (args.out + ".jsonl")
        with open(jl, "w") as fh:
            for r in pts:
                fh.write(json.dumps(_jsonable(r)) + "\n")
        print(f"[out] {jl}", flush=True)

    # 0 = segment finished.  7 = ran clean but did not reach eps_d_end, so
    # run_mc_line.sh should chain another task from the checkpoint.  A job
    # killed by the walltime never gets here at all, which is why the shell
    # also treats "no .npz but a checkpoint exists" as "chain".
    return 0 if (res["complete"] or not pts) else 7


if __name__ == "__main__":
    raise SystemExit(main())
