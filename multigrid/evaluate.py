# multigrid/evaluate.py
# Score any controller on a grid's days by running it in the batched env.
#
# Controllers:
#   policy   : anything with .act(obs) -> action (RL agents, NoOp, Rule-Based)
#   lp_oracle: LP planned on the *actual* day (perfect-foresight lower bound)
#   lp_open  : LP planned on the forecast, executed open loop
#   mpc      : shrinking-horizon LP re-planned every hour on the forecast
#
# All controllers on the same (grid, split, noise, seed) see identical days,
# initial SOC and forecast-error draws.

import numpy as np
import torch

from multigrid.torch_env import BatchedMicrogridEnv, T
from multigrid.lp import grid_params, plan_many

SUMMARY_KEYS = ["cost", "energy", "peak", "demand", "unserved_cost", "unserved_kwh",
                "curtailed_kwh", "degradation", "terminal", "throughput", "peak_violations",
                "import_kwh", "export_kwh", "final_soc"]


MAX_DAYS = None   # set by run_multigrid --quick to cap days per evaluation


def eval_env(bank, gid, split="test", legacy_obs=False):
    """Env with one env per day of `gid` in `split`."""
    k = bank.grid_ids.index(gid)
    days = bank.day_indices(k, split)
    if MAX_DAYS:
        days = days[:MAX_DAYS]
    env = BatchedMicrogridEnv(bank, n_envs=len(days), split=split, legacy_obs=legacy_obs,
                              scale_reward=False)
    return env, days


def _reset(env, days, noise, seed, init_soc):
    env.gen.manual_seed(10_000 + seed)
    return env.reset(day_idx=days, init_soc=torch.full((len(days),), init_soc, device=env.device),
                     noise_level=noise)


def run_policy(policy, bank, gid, split="test", noise=0.0, seed=0, init_soc=0.5):
    env, days = eval_env(bank, gid, split)
    obs = _reset(env, days, noise, seed, init_soc)
    with torch.no_grad():
        for t in range(T):
            obs, _, _, _ = env.step(policy.act(obs))
    return env.summary(), env


def _jobs(env, prm, idx_rows, h, soc, prev, runmax, source):
    L, P = getattr(env, source + "load"), getattr(env, source + "pv")
    PI, PE = getattr(env, source + "pimp"), getattr(env, source + "pexp")
    L, P, PI, PE = (x[:, h:].cpu().numpy().astype(float) for x in (L, P, PI, PE))
    return [(prm, L[i], P[i], PI[i], PE[i], float(soc[i]), float(prev[i]), float(runmax[i]))
            for i in idx_rows]


def run_lp(bank, gid, mode, pool=None, split="test", noise=0.0, seed=0, init_soc=0.5):
    """mode in {'lp_oracle', 'lp_open', 'mpc'}."""
    env, days = eval_env(bank, gid, split)
    _reset(env, days, noise, seed, init_soc)
    prm = grid_params(bank.configs[gid])
    N = env.N
    pbat = env.p["p_bat_max"]
    rows = range(N)

    if mode in ("lp_oracle", "lp_open"):
        src = "a_" if mode == "lp_oracle" else "f_"
        zeros = np.zeros(N)
        plans = plan_many(pool, _jobs(env, prm, rows, 0, env.soc.cpu().numpy(), zeros, zeros, src))
        plan = torch.as_tensor(np.stack(plans), dtype=torch.float32, device=env.device)
        for t in range(T):
            env.step(plan[:, t] / pbat)
    elif mode == "mpc":
        for t in range(T):
            jobs = _jobs(env, prm, rows, t, env.soc.cpu().numpy(), env.prev_bp.cpu().numpy(),
                         env.run_max_imp.cpu().numpy(), "f_")
            plans = plan_many(pool, jobs)
            first = torch.as_tensor([p[0] for p in plans], dtype=torch.float32, device=env.device)
            env.step(first / pbat)
    else:
        raise ValueError(mode)
    return env.summary(), env


def summarize(s):
    out = {k + "_mean": float(np.mean(s[k])) for k in SUMMARY_KEYS}
    out["cost_std"] = float(np.std(s["cost"]))
    out["n_days"] = int(len(s["cost"]))
    return out
