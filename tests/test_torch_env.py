# tests/test_torch_env.py
# The batched torch env must reproduce microgrid_env.MicrogridEnv on G1
# (all new features off), and the new constraints must behave physically.
#
#   python -m pytest tests -q        or        python tests/test_torch_env.py

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from microgrid_env import MicrogridEnv, MicrogridConfig
from multigrid.torch_env import DayBank, BatchedMicrogridEnv, T


def _rollout_numpy(solar, load, price, actions, init_soc):
    env = MicrogridEnv(solar, load, price, config=MicrogridConfig())
    obs, _ = env.reset(options={"initial_soc": init_soc})
    obs_seq, rew = [obs], []
    for a in actions:
        obs, r, term, trunc, info = env.step(np.array([a]))
        obs_seq.append(obs); rew.append(r)
    return np.array(obs_seq), np.array(rew), info["total_cost"]


def test_g1_matches_numpy_env(n_days=100, seed=0):
    rng = np.random.default_rng(seed)
    bank = DayBank(["G1"], device="cpu")
    days = rng.choice(len(bank.load), n_days, replace=False)
    socs = rng.uniform(0.2, 0.8, n_days)
    acts = rng.uniform(-1, 1, (n_days, T))
    # include saturating actions to exercise ramp and SOC limits
    acts[: n_days // 4] = np.sign(acts[: n_days // 4])

    env = BatchedMicrogridEnv(bank, n_envs=n_days, legacy_obs=True, scale_reward=False,
                              legacy_soc_limits=True)
    obs = env.reset(day_idx=torch.as_tensor(days), init_soc=torch.as_tensor(socs, dtype=torch.float32))
    t_obs, t_rew = [obs.numpy()], []
    for t in range(T):
        obs, r, done, _ = env.step(torch.as_tensor(acts[:, t], dtype=torch.float32))
        t_obs.append(obs.numpy()); t_rew.append(r.numpy())
    t_obs = np.stack(t_obs, 1)
    t_rew = np.stack(t_rew, 1)
    t_cost = env.summary()["cost"]

    worst = 0.0
    for i, d in enumerate(days):
        solar = bank.pv[d].numpy(); load = bank.load[d].numpy(); price = bank.p_imp[d].numpy()
        n_obs, n_rew, n_cost = _rollout_numpy(solar, load, price, acts[i], float(np.float32(socs[i])))
        np.testing.assert_allclose(t_obs[i, :T], n_obs[:T], atol=2e-4, err_msg="obs day %d" % d)
        np.testing.assert_allclose(t_rew[i], n_rew, atol=2e-4, rtol=1e-4, err_msg="reward day %d" % d)
        worst = max(worst, abs(t_cost[i] - n_cost))
        assert abs(t_cost[i] - n_cost) < 1e-3 * max(1.0, abs(n_cost)), (d, t_cost[i], n_cost)
    print("G1 equivalence OK over %d days, worst |cost diff| = %.2e $" % (n_days, worst))


def _synthetic_bank(cfg, load, pv, price):
    arr = {"load": load[None].astype(np.float32), "pv": pv[None].astype(np.float32),
           "price_import": price[None].astype(np.float32),
           "price_export": (price * cfg.export_ratio)[None].astype(np.float32),
           "split": np.zeros(1, int)}
    return DayBank([cfg.gid], configs={cfg.gid: cfg}, arrays={cfg.gid: arr})


def test_caps_and_demand_charge():
    from dataclasses import replace
    from multigrid.grids import GRIDS
    cfg = replace(GRIDS["G4"], gid="X", import_cap_kw=10.0, export_cap_kw=2.0,
                  demand_charge_per_kw_day=1.0)
    load = np.full(T, 12.0); pv = np.zeros(T); pv[10:14] = 30.0
    bank = _synthetic_bank(cfg, load, pv, np.full(T, 0.1))
    env = BatchedMicrogridEnv(bank, n_envs=1, scale_reward=False)
    env.reset(day_idx=torch.tensor([0]), init_soc=torch.tensor([0.1]))   # empty battery
    for t in range(T):
        env.step(torch.tensor([1.0 if t < 3 else 0.0]))                   # try to charge
    s = env.summary()
    grid = env.traj["grid"][0].numpy()
    assert grid.max() <= 10.0 + 1e-5, "import cap violated"
    assert grid.min() >= -2.0 - 1e-5, "export cap violated"
    # charging is curbed (not load shed) when the cap binds: hours 0-2 shed only 2 kW
    np.testing.assert_allclose(env.traj["bat"][0, :3].numpy(), 0.0, atol=1e-6)
    assert s["unserved_kwh"][0] > 0 and s["curtailed_kwh"][0] > 0
    np.testing.assert_allclose(s["demand"][0], 1.0 * grid.max(), rtol=1e-5)
    print("caps / demand charge OK: shed %.1f kWh, curtailed %.1f kWh"
          % (s["unserved_kwh"][0], s["curtailed_kwh"][0]))


if __name__ == "__main__":
    test_g1_matches_numpy_env()
    test_caps_and_demand_charge()
