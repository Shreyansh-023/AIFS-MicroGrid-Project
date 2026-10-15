# watch_training.py
# Live progress for a running main.py training job.
#
# The training process buffers its stdout, so its log stays empty until it
# exits. This infers progress from what is observable instead: elapsed time,
# which model files have appeared, and the measured throughput of this machine.
#
#   python watch_training.py            # refresh every 30 s
#   python watch_training.py --once     # print one snapshot and exit

import argparse
import os
import sys
import time
from datetime import datetime, timedelta

RESULTS = "results"

# Stages in the order main.py runs them, with their timestep budgets.
STAGES = [
    ("SAC", "sac_model.zip", 200_000),
    ("R-SAC", "rsac_model.zip", 200_000),
    ("PPO", "ppo_model.zip", 200_000),
]

# Measured on this machine (RTX 3050, SAC with train_freq=1).
# PPO is on-policy and runs substantially faster than the SAC variants.
STEPS_PER_SEC = {"SAC": 79.0, "R-SAC": 72.0, "PPO": 210.0}

BAR_W = 34


def find_job():
    """Return (pid, start_time) of the running main.py, or None."""
    try:
        import psutil
    except ImportError:
        return None

    hits = []
    for p in psutil.process_iter(["pid", "name", "cmdline", "create_time"]):
        try:
            if "python" not in (p.info["name"] or "").lower():
                continue
            cmd = " ".join(p.info["cmdline"] or [])
            if "main.py" not in cmd or "watch_training" in cmd:
                continue
            # Several processes can match (shell wrapper + the real worker).
            # The one actually training has burned the most CPU.
            hits.append((p.cpu_times().user + p.cpu_times().system,
                         p.info["pid"],
                         datetime.fromtimestamp(p.info["create_time"])))
        except Exception:
            continue

    if not hits:
        return None
    _, pid, started = max(hits)
    return pid, started


def stage_done(fname):
    return os.path.exists(os.path.join(RESULTS, fname))


def mtime(fname):
    path = os.path.join(RESULTS, fname)
    return datetime.fromtimestamp(os.path.getmtime(path)) if os.path.exists(path) else None


def bar(frac, width=BAR_W):
    frac = max(0.0, min(1.0, frac))
    fill = int(round(frac * width))
    return "#" * fill + "." * (width - fill)


def fmt_dur(seconds):
    if seconds < 0:
        seconds = 0
    td = timedelta(seconds=int(seconds))
    h, rem = divmod(int(td.total_seconds()), 3600)
    m, s = divmod(rem, 60)
    return f"{h}h {m:02d}m" if h else f"{m}m {s:02d}s"


def eta_for(name, budget):
    return budget / STEPS_PER_SEC[name]


def snapshot():
    job = find_job()
    now = datetime.now()

    print("\n" + "=" * 62)
    print("  MICROGRID RL - TRAINING MONITOR")
    print("=" * 62)

    if job:
        pid, started = job
        elapsed = (now - started).total_seconds()
        print(f"  status   : RUNNING   pid {pid}")
        print(f"  started  : {started:%H:%M:%S}")
        print(f"  elapsed  : {fmt_dur(elapsed)}")
    else:
        started = None
        elapsed = 0.0
        if all(stage_done(f) for _, f, _ in STAGES):
            print("  status   : COMPLETE — all three models saved")
        else:
            print("  status   : NOT RUNNING (no main.py process found)")
            print("             partial results below; restart with: python main.py")

    print("-" * 62)

    # Work out which stage is active: the first one without a saved model.
    active_idx = next(
        (i for i, (_, f, _) in enumerate(STAGES) if not stage_done(f)), len(STAGES)
    )

    # Time already accounted for by finished stages.
    done_secs = sum(eta_for(n, b) for (n, _, b) in STAGES[:active_idx])
    total_secs = sum(eta_for(n, b) for (n, _, b) in STAGES)

    for i, (name, fname, budget) in enumerate(STAGES):
        if i < active_idx:
            ts = mtime(fname)
            stamp = f"saved {ts:%H:%M:%S}" if ts else "saved"
            print(f"  [{i+1}/3] {name:<6} [{bar(1.0)}] 100%  {stamp}")
        elif i == active_idx and job:
            # Progress within the current stage, from time spent in it.
            in_stage = max(0.0, elapsed - done_secs)
            frac = min(0.999, in_stage / eta_for(name, budget))
            steps = int(frac * budget)
            print(
                f"  [{i+1}/3] {name:<6} [{bar(frac)}] {frac*100:4.1f}%  "
                f"~{steps:,}/{budget:,} steps"
            )
        else:
            print(f"  [{i+1}/3] {name:<6} [{bar(0.0)}]   0%  queued")

    print("-" * 62)

    all_saved = all(stage_done(f) for _, f, _ in STAGES)

    if job and not all_saved:
        remaining = max(0.0, total_secs - elapsed)
        finish = now + timedelta(seconds=remaining)
        print(f"  training done (est) : {fmt_dur(remaining)}  ->  {finish:%H:%M}")
        print("  then: evaluation + 13 plots + robustness sweep (~10 min)")
        print(f"  all done (est)      : {finish + timedelta(minutes=10):%H:%M}")
        print()
        print("  Estimates assume measured throughput; actual pace may vary.")
    elif job and all_saved:
        # Models are written; the job is in the evaluate/plot/sweep tail.
        print("  All three models saved. Now running:")
        print("  evaluation -> 13 plots -> robustness sweep  (~10 min)")
    elif all_saved:
        print("  Next: python export_web_data.py   (refresh the dashboard)")

    print("=" * 62)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true", help="print one snapshot and exit")
    ap.add_argument("--interval", type=int, default=30, help="refresh seconds")
    args = ap.parse_args()

    if args.once:
        snapshot()
        return

    try:
        while True:
            os.system("cls" if os.name == "nt" else "clear")
            snapshot()
            print(f"\n  refreshing every {args.interval}s — Ctrl+C to stop")
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\nstopped watching (training continues in the background)\n")


if __name__ == "__main__":
    main()
