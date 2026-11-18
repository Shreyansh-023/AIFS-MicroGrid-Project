# multigrid/agents/baselines.py
# Batched versions of the NoOp and Rule-Based agents in rl_agents.py.

import torch

from multigrid.agents.common import Policy


class NoopPolicy(Policy):
    name = "NoOp"

    def act(self, obs):
        return torch.zeros(obs.shape[0], device=obs.device)


class RuleBasedPolicy(Policy):
    """Same heuristic as rl_agents.RuleBasedAgent, on batched observations."""
    name = "Rule-Based"

    def act(self, obs):
        soc = (obs[:, 0] + 1) / 2
        net, price = obs[:, 1], obs[:, 2]
        a = torch.zeros(obs.shape[0], device=obs.device)
        a = torch.where((net < -0.2) & (soc < 0.9), torch.full_like(a, 0.8), a)
        a = torch.where((net > 0.2) & (soc > 0.2) & (price > 0.1), torch.full_like(a, -0.8), a)
        return a


class SelfConsumptionPolicy(Policy):
    """Industry-standard self-consumption controller with end-of-day SOC return.

    Charges with surplus PV, discharges to cover the deficit (both capped by
    the inverter), and over the last `steer_hours` hours steers the battery to
    the target SOC. Reads absolute scales from the extended observation.
    """
    name = "Self-Cons."

    def __init__(self, steer_hours=4, target_soc=0.5):
        self.steer_hours = steer_hours
        self.target = target_soc

    def act(self, obs):
        from multigrid.torch_env import LEGACY_OBS_DIM
        import math
        soc = (obs[:, 0] + 1) / 2
        hour = torch.remainder(torch.atan2(obs[:, 3], obs[:, 4]) / (2 * math.pi) * 24, 24).round()
        net_over_pbat = obs[:, 1] * obs[:, LEGACY_OBS_DIM + 9] * 5.0     # net kW / P_bat
        storage_h = obs[:, LEGACY_OBS_DIM] * 8.0                          # E_max / P_bat
        a = torch.clamp(-net_over_pbat, -1, 1)
        remaining = (24 - hour).clamp(min=1)
        steer = (self.target - soc) * storage_h / remaining
        return torch.where(hour >= 24 - self.steer_hours, torch.clamp(steer, -1, 1), a)
