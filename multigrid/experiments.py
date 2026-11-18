# multigrid/experiments.py
# Training and the four paper experiments. Every evaluated day is appended to
# results/multigrid/daily_results.csv (long format), which all tables, plots
# and statistics are computed from.
#
#   E1  per-grid benchmark   specialists vs LP / MPC / heuristics, 0% error
#   E2  generalist           one policy for all grids vs the specialists
#   E3  generalisation       leave-one-grid-out + cross-grid transfer matrix
#   E4  robustness           forecast-error sweep per grid

import json
import time
from pathlib import Path
from dataclasses import replace

import numpy as np
import pandas as pd
import torch

from multigrid.grids import GRID_IDS
from multigrid.torch_env import DayBank, BatchedMicrogridEnv
from multigrid.agents.ppo import PPOAgent, PPOConfig
from multigrid.agents.sac import SACAgent, SACConfig
from multigrid.agents.baselines import NoopPolicy, RuleBasedPolicy, SelfConsumptionPolicy
from multigrid import evaluate as ev

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "multigrid"
CKPT = RESULTS / "checkpoints"
DAILY = RESULTS / "daily_results.csv"

# algorithm variants: base hyper-parameters + training noise (R- = domain randomised)
ALGOS = {
    "ppo":  {"kind": "ppo", "label": "PPO",   "noise": 0.0, "variable": False},
    "rppo": {"kind": "ppo", "label": "R-PPO", "noise": 0.3, "variable": True},
    "sac":  {"kind": "sac", "label": "SAC",   "noise": 0.0, "variable": False},
    "rsac": {"kind": "sac", "label": "R-SAC", "noise": 0.3, "variable": True},
}
# tuned on G1/G3 validation days (see README "Multi-grid study")
PPO_BASE = PPOConfig(n_envs=512, total_steps=10_000_000, minibatch=2048, epochs=10, init_log_std=-0.5)
SAC_BASE = SACConfig(n_envs=128, total_steps=6_000_000, updates_per_step=4, batch_size=1024)
GENERALIST_MULT = 1.5       # generalists get this many times the specialist budget
BUDGET_SCALE = 1.0          # multiplies total_steps (run_multigrid --budget-scale)

HEURISTICS = {"NoOp": NoopPolicy(), "Rule-Based": RuleBasedPolicy(), "Self-Cons.": SelfConsumptionPolicy()}
LP_MODES = {"LP (oracle)": "lp_oracle", "LP (open loop)": "lp_open", "MPC": "mpc"}


def device():
    return "cuda" if torch.cuda.is_available() else "cpu"


_BANK = None


def global_bank():
    global _BANK
    if _BANK is None:
        _BANK = DayBank(GRID_IDS, device=device())
    return _BANK


# ------------------------------------------------------------------ train --

def ckpt_path(algo, tag, seed):
    return CKPT / ("%s_%s_s%d.pt" % (algo, tag, seed))


def _make_agent(kind, obs_dim, steps_mult=1.0, n_train_grids=1):
    if kind == "ppo":
        cfg = replace(PPO_BASE, total_steps=int(PPO_BASE.total_steps * BUDGET_SCALE * steps_mult))
        return PPOAgent(obs_dim, cfg, device())
    cfg = replace(SAC_BASE, total_steps=int(SAC_BASE.total_steps * BUDGET_SCALE * steps_mult))
    return SACAgent(obs_dim, cfg, device())


def load_agent(algo, tag, seed):
    st = torch.load(ckpt_path(algo, tag, seed), map_location=device(), weights_only=False)
    cls = PPOAgent if st["algo"] == "ppo" else SACAgent
    agent = cls.from_state(st, device())
    agent.name = ALGOS[algo]["label"]
    return agent


def train(algo, train_grids, tag, seed, force=False, verbose=True):
    """Train (or load) one agent. tag: grid id, 'ALL', or 'LOO-Gx'."""
    path = ckpt_path(algo, tag, seed)
    if path.exists() and not force:
        return load_agent(algo, tag, seed)
    spec = ALGOS[algo]
    torch.manual_seed(seed); np.random.seed(seed)
    bank = DayBank(train_grids, device=device())
    # generalists see more variety: give them proportionally more samples
    mult = 1.0 if len(train_grids) == 1 else GENERALIST_MULT
    agent = _make_agent(spec["kind"], _obs_dim(bank), mult, len(train_grids))
    env = BatchedMicrogridEnv(bank, agent.cfg.n_envs, split="train", noise_level=spec["noise"],
                              variable_noise=spec["variable"], correlation=0.9, seed=seed)

    noop_val = {g: float(ev.run_policy(NoopPolicy(), bank, g, split="val")[0]["cost"].mean())
                for g in train_grids}

    def val_fn(a):
        # selection score: mean over training grids of cost / NoOp cost (scale-free)
        costs = {g: float(ev.run_policy(a, bank, g, split="val")[0]["cost"].mean()) for g in train_grids}
        out = {g: round(c, 2) for g, c in list(costs.items())[:3]}
        out["_score"] = round(float(np.mean([costs[g] / noop_val[g] for g in train_grids])), 4)
        return out

    if verbose:
        print("\n[train] %s on %s (%s) seed %d, %d steps" % (spec["label"], tag, ",".join(train_grids),
                                                              seed, agent.cfg.total_steps), flush=True)
    t0 = time.time()
    log_every = 100 if spec["kind"] == "ppo" else 400
    agent.train(env, log_every=log_every, val_fn=val_fn, verbose=verbose)
    st = agent.state()
    st["train_time_s"] = time.time() - t0
    st["train_grids"] = list(train_grids)
    CKPT.mkdir(parents=True, exist_ok=True)
    torch.save(st, path)
    agent.name = spec["label"]
    return agent


def _obs_dim(bank):
    return BatchedMicrogridEnv(bank, 1).obs_dim


# ---------------------------------------------------------------- records --

def _rows(exp, gid, method, seed, noise, scen, summary, train_tag=""):
    n = len(summary["cost"])
    df = pd.DataFrame({k: summary[k] for k in ev.SUMMARY_KEYS})
    df.insert(0, "day", summary["day_idx"])
    for col, val in (("train_tag", train_tag), ("scenario", scen), ("noise", noise), ("seed", seed),
                     ("method", method), ("grid", gid), ("experiment", exp)):
        df.insert(0, col, [val] * n)
    return df


class Recorder:
    def __init__(self, path=DAILY):
        self.path = path
        self.frames = []
        self.done = set()
        if path.exists():
            old = pd.read_csv(path)
            old["train_tag"] = old["train_tag"].fillna("").astype(str)
            self.frames.append(old)
            self.done = set(map(tuple, old[["experiment", "grid", "method", "seed", "noise", "scenario",
                                            "train_tag"]].drop_duplicates().astype(str).values))

    def has(self, *key):
        return tuple(str(k) for k in key) in self.done

    def add(self, df):
        self.frames.append(df)
        k = df.iloc[0]
        self.done.add(tuple(str(k[c]) for c in ("experiment", "grid", "method", "seed", "noise",
                                                "scenario", "train_tag")))

    def save(self):
        RESULTS.mkdir(parents=True, exist_ok=True)
        pd.concat(self.frames, ignore_index=True).to_csv(self.path, index=False)


def _eval_baselines(rec, exp, gid, noise, scen, pool, lp_modes):
    bank = global_bank()
    for name, pol in HEURISTICS.items():
        if not rec.has(exp, gid, name, 0, noise, scen, ""):
            s, _ = ev.run_policy(pol, bank, gid, noise=noise, seed=scen)
            rec.add(_rows(exp, gid, name, 0, noise, scen, s))
    for name in lp_modes:
        if not rec.has(exp, gid, name, 0, noise, scen, ""):
            s, _ = ev.run_lp(bank, gid, LP_MODES[name], pool=pool, noise=noise, seed=scen)
            rec.add(_rows(exp, gid, name, 0, noise, scen, s))


def _eval_agent(rec, exp, gid, agent, label, seed, noise, scen, tag):
    if not rec.has(exp, gid, label, seed, noise, scen, tag):
        s, _ = ev.run_policy(agent, global_bank(), gid, noise=noise, seed=scen)
        rec.add(_rows(exp, gid, label, seed, noise, scen, s, tag))


# ------------------------------------------------------------ experiments --

def e1(grids, algos, seeds, pool, rec, train_only=False):
    for g in grids:
        for a in algos:
            for s in seeds:
                agent = train(a, [g], g, s)
                if not train_only:
                    _eval_agent(rec, "E1", g, agent, ALGOS[a]["label"], s, 0.0, 0, g)
        if not train_only:
            _eval_baselines(rec, "E1", g, 0.0, 0, pool, ["LP (oracle)", "MPC"])
            rec.save()


def e2(grids, algos, seeds, pool, rec, train_only=False):
    for a in algos:
        for s in seeds:
            agent = train(a, list(grids), "ALL", s)
            if train_only:
                continue
            for g in grids:
                _eval_agent(rec, "E2", g, agent, "Generalist " + ALGOS[a]["label"], s, 0.0, 0, "ALL")
    rec.save()


def e3(grids, algos, seeds, pool, rec, train_only=False):
    # leave one grid out
    for a in algos:
        for s in seeds:
            for g in grids:
                others = [x for x in grids if x != g]
                agent = train(a, others, "LOO-" + g, s)
                if not train_only:
                    _eval_agent(rec, "E3", g, agent, "Zero-shot " + ALGOS[a]["label"], s, 0.0, 0, "LOO-" + g)
            # transfer matrix: specialist trained on i, tested on j
            if train_only:
                continue
            for gi in grids:
                agent = train(a, [gi], gi, s)
                for gj in grids:
                    _eval_agent(rec, "E3T", gj, agent, ALGOS[a]["label"], s, 0.0, 0, gi)
        rec.save()


def e4(grids, algos, seeds, pool, rec, noise_levels=(0.0, 0.05, 0.1, 0.2, 0.3), n_scen=3,
       train_only=False):
    for g in grids:
        agents = []
        for a in algos:
            for s in seeds:
                agents.append((ALGOS[a]["label"], s, train(a, [g], g, s), g))
        if train_only:
            continue
        for noise in noise_levels:
            for sc in range(n_scen):
                _eval_baselines(rec, "E4", g, noise, sc, pool, ["LP (oracle)", "LP (open loop)", "MPC"])
                for label, s, agent, tag in agents:
                    _eval_agent(rec, "E4", g, agent, label, s, noise, sc, tag)
            print("  [E4] %s noise %.2f done" % (g, noise), flush=True)
            rec.save()
