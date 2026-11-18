# run_multigrid.py - multi-grid study: data -> train -> experiments -> report
#
#   python run_multigrid.py --quick                     smoke test (~5 min)
#   python run_multigrid.py                             all experiments, defaults
#   python run_multigrid.py --experiments E1 --grids G1,G3 --algos ppo,sac --seeds 0,1,2
#   python run_multigrid.py --train-only                train every needed agent
#   python run_multigrid.py --report-only               rebuild tables / figures
#
# Checkpoints and finished evaluations are reused, so an interrupted run can be
# restarted with the same command.

import argparse
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")


def parse():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--quick", action="store_true", help="tiny budgets on G1,G3 into results/multigrid_quick")
    ap.add_argument("--experiments", default="E1,E2,E3,E4")
    ap.add_argument("--grids", default="all")
    ap.add_argument("--algos", default="sac,ppo", help="E1/E2 algorithms; E3 uses the first one")
    ap.add_argument("--generalist-algos", default="sac", help="algorithms for E2/E3")
    ap.add_argument("--robust-algos", default="sac,rsac,ppo,rppo", help="agents compared in E4")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--e3-seeds", default="0,1", help="seeds for leave-one-grid-out runs")
    ap.add_argument("--e4-seeds", default="0,1", help="seeds for the robustness sweep")
    ap.add_argument("--noise-levels", default="0,0.05,0.1,0.2,0.3")
    ap.add_argument("--scenarios", type=int, default=3, help="forecast-error draws per noise level")
    ap.add_argument("--budget-scale", type=float, default=1.0, help="scale all training steps")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--train-only", action="store_true")
    ap.add_argument("--report-only", action="store_true")
    ap.add_argument("--skip-data", action="store_true", help="do not (re)build missing datasets")
    return ap.parse_args()


def main():
    args = parse()
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from multigrid import experiments as X
    from multigrid import evaluate as ev
    from multigrid import report
    from multigrid.grids import GRID_IDS
    from multigrid.data import build_dataset

    grids = GRID_IDS if args.grids == "all" else args.grids.split(",")
    algos = args.algos.split(",")
    robust = args.robust_algos.split(",")
    gen_algos = args.generalist_algos.split(",")
    seeds = [int(s) for s in args.seeds.split(",")]
    e3_seeds = [int(s) for s in args.e3_seeds.split(",")]
    e4_seeds = [int(s) for s in args.e4_seeds.split(",")]
    noise = [float(x) for x in args.noise_levels.split(",")]
    exps = args.experiments.split(",")
    X.BUDGET_SCALE = args.budget_scale
    n_scen = args.scenarios

    if args.quick:
        grids = ["G1", "G3"] if args.grids == "all" else grids
        algos, robust, gen_algos = ["ppo"], ["ppo", "rppo"], ["ppo"]
        seeds = e3_seeds = e4_seeds = [0]
        noise, n_scen = [0.0, 0.2], 1
        X.BUDGET_SCALE = 0.05 * args.budget_scale
        ev.MAX_DAYS = 12
        X.RESULTS = Path(__file__).resolve().parent / "results" / "multigrid_quick"
        X.CKPT = X.RESULTS / "checkpoints"
        X.DAILY = X.RESULTS / "daily_results.csv"
        report.RESULTS = X.RESULTS

    out = X.RESULTS
    out.mkdir(parents=True, exist_ok=True)
    if args.report_only:
        report.build_report(out=out)
        return

    # data
    if not args.skip_data:
        for g in GRID_IDS if ("E2" in exps or "E3" in exps) else grids:
            if not (build_dataset.PROC / ("%s.npz" % g)).exists():
                build_dataset.build_grid(g)

    t0 = time.time()
    rec = X.Recorder(X.DAILY)
    print("device: %s | grids %s | algos %s | seeds %s | budget x%.2f" % (
        X.device(), grids, algos, seeds, X.BUDGET_SCALE), flush=True)
    with mp.get_context("spawn").Pool(args.workers) as pool:
        if "E1" in exps:
            print("\n=== E1 per-grid benchmark ===", flush=True)
            X.e1(grids, algos, seeds, pool, rec, args.train_only)
        if "E2" in exps:
            print("\n=== E2 generalist ===", flush=True)
            X.e2(grids, gen_algos, seeds, pool, rec, args.train_only)
        if "E3" in exps:
            print("\n=== E3 generalisation ===", flush=True)
            X.e3(grids, gen_algos, e3_seeds, pool, rec, args.train_only)
        if "E4" in exps:
            print("\n=== E4 robustness ===", flush=True)
            X.e4(grids, robust, e4_seeds, pool, rec, noise, n_scen, args.train_only)
    if not args.train_only:
        rec.save()
        report.build_report(out=out)
    print("\ntotal time %.1f min" % ((time.time() - t0) / 60))


if __name__ == "__main__":
    main()
