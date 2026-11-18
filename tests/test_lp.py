# tests/test_lp.py
# The generalised LP must (1) reproduce lp_solver.py on G1 when configured the
# same way, (2) be scored identically by the env (objective == executed cost),
# (3) never lose to NoOp / Rule-Based at 0% forecast error.

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lp_solver import MicrogridLPSolver, LPConfig
from multigrid.grids import GRIDS, GRID_IDS
from multigrid.lp import grid_params, solve_schedule
from multigrid.torch_env import DayBank
from multigrid.evaluate import run_lp, run_policy
from multigrid.agents.baselines import NoopPolicy, RuleBasedPolicy


def test_matches_legacy_lp(n=20):
    bank = DayBank(["G1"])
    prm = grid_params(GRIDS["G1"])
    old = MicrogridLPSolver(LPConfig())
    rng = np.random.default_rng(1)
    for d in rng.choice(len(bank.load), n, replace=False):
        L, P, PI = bank.load[d].numpy(), bank.pv[d].numpy(), bank.p_imp[d].numpy()
        soc0 = float(rng.uniform(0.2, 0.8))
        r = old.solve(P, L, PI, initial_soc=soc0)
        c = old.config
        old_obj = (np.sum(r["grid_power"].clip(min=0) * PI) - np.sum((-r["grid_power"]).clip(min=0) * PI * 0.4)
                   + np.sum(r["peak_violations"]) * c.peak_penalty_rate
                   + c.degradation_cost * (c.eta_charge * r["charging_power"].sum()
                                           + r["discharging_power"].sum() / c.eta_discharge))
        new = solve_schedule(prm, L, P, PI, PI * 0.4, soc0, hard_terminal=True, ramp_first=False)
        assert abs(new["objective"] - old_obj) < 1e-4 * max(1, abs(old_obj)), (d, new["objective"], old_obj)
    print("legacy LP reproduced on %d G1 days" % n)


def test_lp_consistent_and_dominant():
    bank = DayBank(GRID_IDS)
    for g in GRID_IDS:
        s_lp, env = run_lp(bank, g, "lp_oracle", split="val")
        prm = grid_params(GRIDS[g])
        # objective of the LP for the first few days vs executed env cost
        for i in range(3):
            d = int(env.day_idx[i])
            r = solve_schedule(prm, bank.load[d].numpy(), bank.pv[d].numpy(),
                               bank.p_imp[d].numpy(), bank.p_exp[d].numpy(), 0.5)
            assert abs(r["objective"] - s_lp["cost"][i]) < 1e-3 * max(1, abs(r["objective"])) + 1e-3, \
                (g, i, r["objective"], s_lp["cost"][i])
        s_noop, _ = run_policy(NoopPolicy(), bank, g, split="val")
        s_rule, _ = run_policy(RuleBasedPolicy(), bank, g, split="val")
        assert np.all(s_lp["cost"] <= s_noop["cost"] + 1e-3), g
        assert np.all(s_lp["cost"] <= s_rule["cost"] + 1e-3), g
        print("%s  LP %9.2f  NoOp %9.2f  Rule %9.2f  $/day  (LP saves %.1f%%)" % (
            g, s_lp["cost"].mean(), s_noop["cost"].mean(), s_rule["cost"].mean(),
            100 * (1 - s_lp["cost"].mean() / s_noop["cost"].mean())))


if __name__ == "__main__":
    test_matches_legacy_lp()
    test_lp_consistent_and_dominant()
