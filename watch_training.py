# watch_training.py
# Live progress for a running training job (main.py or resume_training.py).
#
# When the job has its stdout redirected to a log file, this reads the real
# step counts out of it. Without a log it falls back to inferring progress
# from elapsed time, which model files have appeared, and the measured
# throughput of this machine.
#
#   python watch_training.py                  # refresh every 30 s
#   python watch_training.py --once           # print one snapshot and exit
#   python watch_training.py --log other.log  # read a specific log

import argparse
import os
import re
import sys
import time
from datetime import datetime, timedelta

RESULTS = "results"

# Stages in the order both pipelines run them, with their timestep budgets.
STAGES = [
    ("SAC", "sac_model.zip", 200_000),
    ("R-SAC", "rsac_model.zip", 200_000),
    ("PPO", "ppo_model.zip", 200_000),
]

# Measured on this machine (RTX 3050, SAC with train_freq=1).
# PPO is on-policy and runs substantially faster than the SAC variants.
STEPS_PER_SEC = {"SAC": 79.0, "R-SAC": 72.0, "PPO": 210.0}

# Either script runs the same three stages, so either may be the job.
JOB_SCRIPTS = ("resume_training.py", "main.py")

# Logs to look for when --log is not given; the newest match wins.
LOG_CANDIDATES = ("resume_training.log", "training.log", "main.log")

BAR_W = 34

# Both pipelines label their stages "[n/3]", and TrainingCallback prints
# "Step 38000: Mean reward (last 10): -1.01".
STAGE_RE = re.compile(r"\[(\d)/3\]\s*(.*)")
STEP_RE = re.compile(r"^Step\s+([\d,]+)\s*:")

# What runs after the last model is saved, in the order it is printed.
TAIL_PHASES = [
    ("evaluation", ("Evaluating All Agents", "STEP 4:")),
    ("plots", ("Generating Visualizations", "STEP 5:")),
    ("robustness", ("Robustness Testing", "STEP 6:")),
    ("complete", ("PIPELINE COMPLETE", "SIMULATION COMPLETE")),
]

TAIL_LABELS = {
    "evaluation": "evaluation (100 episodes, every agent)",
    "plots": "13 plots",
    "robustness": "robustness sweep (5 noise levels x 10 scenarios)",
}


def find_job():
    """Return (pid, start_time, script) of the running pipeline, or None."""
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
            if "watch_training" in cmd:
                continue
            script = next((s for s in JOB_SCRIPTS if s in cmd), None)
            if script is None:
                continue
            # Several processes can match (shell wrapper + the real worker).
            # The one actually training has burned the most CPU.
            hits.append((p.cpu_times().user + p.cpu_times().system,
                         p.info["pid"],
                         datetime.fromtimestamp(p.info["create_time"]),
                         script))
        except Exception:
            continue

    if not hits:
        return None
    _, pid, started, script = max(hits)
    return pid, started, script


def find_log(explicit=None):
    """Pick the log to read: the one asked for, else the newest known name."""
    if explicit:
        return explicit if os.path.exists(explicit) else None
    found = [p for p in LOG_CANDIDATES if os.path.exists(p)]
    return max(found, key=os.path.getmtime) if found else None


def read_log(path):
    """Pull the current stage, step count and tail phase out of the job log."""
    state = {"stage": None, "step": 0, "reused": set(), "phase": None, "last": ""}
    try:
        with open(path, "r", errors="replace") as fh:
            lines = fh.readlines()
    except OSError:
        return state

    for raw in lines:
        line = raw.strip()
        if not line:
            continue

        m = STAGE_RE.match(line)
        if m:
            state["stage"] = int(m.group(1)) - 1
            state["step"] = 0
            if "loading saved" in line:
                state["reused"].add(state["stage"])
            continue

        m = STEP_RE.match(line)
        if m:
            state["step"] = int(m.group(1).replace(",", ""))
            state["last"] = line
            continue

        for name, markers in TAIL_PHASES:
            if any(mk in line for mk in markers):
                state["phase"] = name
        state["last"] = line

    return state


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


def stage_start(active_idx, started):
    """When the active stage began training."""
    if started is None:
        return None
    if active_idx <= 0:
        return started
    # The previous stage ended when its model was written - unless that model
    # predates this job, in which case it was reused and cost no time.
    prev_ts = mtime(STAGES[active_idx - 1][1])
    return prev_ts if (prev_ts and prev_ts >= started) else started


def observed_rate(active_idx, log, started, log_path):
    """Steps per second measured from the log, or None if not yet meaningful.

    The hardcoded STEPS_PER_SEC figures assume a quiet machine; the real pace
    depends on what else is running, so prefer a measurement when we have one.
    """
    if not (log and log["step"] and log_path and started is not None):
        return None
    if active_idx >= len(STAGES):
        return None

    begin = stage_start(active_idx, started)
    try:
        # The log is touched on every print, so its mtime is when the last
        # step line landed.
        last = datetime.fromtimestamp(os.path.getmtime(log_path))
    except OSError:
        return None

    span = (last - begin).total_seconds()
    if span < 60 or log["step"] < 4000:
        return None
    return log["step"] / span


def remaining_secs(active_idx, log, elapsed, done_secs, rate=None):
    """Training seconds left, from real step counts when the log has them."""
    if active_idx >= len(STAGES):
        return 0.0

    name, _, budget = STAGES[active_idx]
    expected = STEPS_PER_SEC[name]

    if log and log["step"]:
        steps_left = max(0, budget - min(log["step"], budget))
        left = steps_left / (rate or expected)
    else:
        left = max(0.0, eta_for(name, budget) - max(0.0, elapsed - done_secs))

    # The machine paces every stage the same way, so carry whatever slowdown
    # we measured over to the stages still queued.
    slow = (expected / rate) if rate else 1.0
    return left + sum(eta_for(n, b) * slow for (n, _, b) in STAGES[active_idx + 1:])


def snapshot(log_path=None):
    job = find_job()
    now = datetime.now()
    log = read_log(log_path) if log_path else None

    print("\n" + "=" * 62)
    print("  MICROGRID RL - TRAINING MONITOR")
    print("=" * 62)

    all_saved = all(stage_done(f) for _, f, _ in STAGES)

    if job:
        pid, started, script = job
        elapsed = (now - started).total_seconds()
        print(f"  status   : RUNNING   pid {pid}  ({script})")
        print(f"  started  : {started:%H:%M:%S}")
        print(f"  elapsed  : {fmt_dur(elapsed)}")
    else:
        started = None
        elapsed = 0.0
        if all_saved:
            print("  status   : COMPLETE - all three models saved")
        else:
            print("  status   : NOT RUNNING (no pipeline process found)")
            print("             partial results below; pick up where it stopped:")
            print("             python resume_training.py")

    if log_path:
        print(f"  log      : {log_path}")

    print("-" * 62)

    # First stage with no saved model.
    file_idx = next(
        (i for i, (_, f, _) in enumerate(STAGES) if not stage_done(f)), len(STAGES)
    )

    phase = log["phase"] if log else None

    # The log knows which stage is really active. A stage it reports as reused
    # is not being worked on, so fall through to the first unfinished one.
    if log and log["stage"] is not None and phase is None:
        active_idx = log["stage"]
        if active_idx in log["reused"] or stage_done(STAGES[active_idx][1]):
            active_idx = max(active_idx, file_idx)
    else:
        active_idx = file_idx

    def was_reused(i, fname):
        """True if the job loaded this model instead of training it."""
        if log and i in log["reused"]:
            return True
        ts = mtime(fname)
        return bool(started and ts and ts < started)

    # Time the finished stages account for. A reused model cost nothing.
    done_secs = sum(
        eta_for(n, b)
        for i, (n, f, b) in enumerate(STAGES)
        if i < active_idx and not was_reused(i, f)
    )

    for i, (name, fname, budget) in enumerate(STAGES):
        if i < active_idx:
            ts = mtime(fname)
            if was_reused(i, fname):
                stamp = "reused" + (f" (saved {ts:%H:%M:%S})" if ts else "")
            else:
                stamp = f"saved {ts:%H:%M:%S}" if ts else "saved"
            print(f"  [{i+1}/3] {name:<6} [{bar(1.0)}] 100%  {stamp}")
        elif i == active_idx and job:
            if log and log["step"]:
                steps = min(log["step"], budget)
                frac = min(0.999, steps / budget)
                detail = f"{steps:,}/{budget:,} steps"
            else:
                # No log to read: infer from time spent in this stage.
                in_stage = max(0.0, elapsed - done_secs)
                frac = min(0.999, in_stage / eta_for(name, budget))
                steps = int(frac * budget)
                detail = f"~{steps:,}/{budget:,} steps"
            print(f"  [{i+1}/3] {name:<6} [{bar(frac)}] {frac*100:4.1f}%  {detail}")
        else:
            print(f"  [{i+1}/3] {name:<6} [{bar(0.0)}]   0%  queued")

    print("-" * 62)

    if job and phase == "complete":
        print("  pipeline finished - the process is on its way out.")
        print("  Next: python export_web_data.py   (refresh the dashboard)")
    elif job and phase:
        print("  training done. now running: %s" % TAIL_LABELS[phase])
        print("  the tail stages take ~10 min in total")
    elif job and not all_saved:
        rate = observed_rate(active_idx, log, started, log_path)
        if rate:
            name = STAGES[active_idx][0]
            print("  measured pace       : %.0f steps/s  (%s budgeted at %.0f)"
                  % (rate, name, STEPS_PER_SEC[name]))
        remaining = remaining_secs(active_idx, log, elapsed, done_secs, rate)
        finish = now + timedelta(seconds=remaining)
        print(f"  training done (est) : {fmt_dur(remaining)}  ->  {finish:%H:%M}")
        print("  then: evaluation + 13 plots + robustness sweep (~10 min)")
        print(f"  all done (est)      : {finish + timedelta(minutes=10):%H:%M}")
        print()
        if rate:
            print("  Estimate uses the pace measured so far; it moves as that does.")
        else:
            print("  Estimates assume budgeted throughput; actual pace may vary.")
    elif job and all_saved:
        # Models are written; the job is in the evaluate/plot/sweep tail.
        print("  All three models saved. Now running:")
        print("  evaluation -> 13 plots -> robustness sweep  (~10 min)")
    elif all_saved:
        print("  Next: python export_web_data.py   (refresh the dashboard)")
    else:
        # Nothing running and models missing: the job stopped partway.
        nxt = STAGES[file_idx][0] if file_idx < len(STAGES) else None
        print("  stopped before %s. resume_training.py picks up from here and" % nxt)
        print("  keeps the models already saved.")

    if log and log["last"]:
        print("-" * 62)
        print("  log tail : %s" % log["last"][:52])

    print("=" * 62)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true", help="print one snapshot and exit")
    ap.add_argument("--interval", type=int, default=30, help="refresh seconds")
    ap.add_argument("--log", help="job log to read step counts from")
    args = ap.parse_args()

    if args.once:
        snapshot(find_log(args.log))
        return

    try:
        while True:
            os.system("cls" if os.name == "nt" else "clear")
            # Resolved each pass: the log may only appear once the job starts.
            snapshot(find_log(args.log))
            print(f"\n  refreshing every {args.interval}s - Ctrl+C to stop")
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\nstopped watching (training continues in the background)\n")


if __name__ == "__main__":
    main()
