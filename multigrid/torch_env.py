# multigrid/torch_env.py
# GPU-batched microgrid environment: N independent microgrids stepped at once.
#
# Physics follow microgrid_env.MicrogridEnv.step exactly (ramp limit, SOC
# window, charge/discharge efficiency, TOU energy cost, soft peak penalty,
# degradation, terminal SOC penalty) and add, per grid:
#   * separate import / export price arrays (real-time prices)
#   * incremental demand charge on the running daily max import
#   * import cap  -> charging reduced first, remaining excess is shed at VoLL
#   * export cap  -> discharging reduced first, remaining excess PV curtailed
# With these switched off (G1) the env reproduces MicrogridEnv to float32
# precision (tests/test_torch_env.py).
#
# All envs run synchronised 24-step episodes (one day each), so a rollout of
# 24 steps is exactly one episode for every env.

import json
import math
from pathlib import Path

import numpy as np
import torch

from multigrid.grids import GRIDS, grid_descriptor, finite, DESCRIPTOR_DIM

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"

T = 24
HIST = 4
BIG = 1e9
SPLITS = {"train": 0, "val": 1, "test": 2}
LEGACY_OBS_DIM = 7 + HIST + 5 * T          # 131, identical to MicrogridEnv
EXTRA_DIM = 3
OBS_DIM = LEGACY_OBS_DIM + DESCRIPTOR_DIM + EXTRA_DIM

PARAM_KEYS = ["e_max", "e_min_ratio", "p_bat_max", "eta_charge", "eta_discharge",
              "ramp_rate", "degradation_cost_per_kwh", "target_final_soc",
              "terminal_penalty_per_kwh", "demand_charge_per_kw_day", "p_grid_peak",
              "peak_penalty_rate", "import_cap_kw", "export_cap_kw", "voll"]


class DayBank:
    """All grids' day profiles as tensors on one device."""

    def __init__(self, grid_ids, device="cpu", configs=None, arrays=None):
        self.grid_ids = list(grid_ids)
        self.device = torch.device(device)
        configs = configs or {}
        self.configs = {g: configs[g] if g in configs else GRIDS[g] for g in self.grid_ids}
        loads, pvs, pis, pes, owner, split, self.meta, self.dates = [], [], [], [], [], [], {}, []
        for k, g in enumerate(self.grid_ids):
            if arrays is not None and g in arrays:
                d = arrays[g]
                meta = {"mean_load_kw": float(np.mean(d["load"])),
                        "mean_import_price": float(np.mean(d["price_import"]))}
            else:
                d = np.load(PROC / ("%s.npz" % g))
                meta = json.loads((PROC / ("%s_meta.json" % g)).read_text())
            self.meta[g] = meta
            n = len(d["load"])
            loads.append(d["load"]); pvs.append(d["pv"])
            pis.append(d["price_import"]); pes.append(d["price_export"])
            owner.append(np.full(n, k)); split.append(d["split"] if "split" in d else np.zeros(n, int))
            self.dates += list(d["dates"]) if "dates" in d else [""] * n
        f = lambda a: torch.as_tensor(np.concatenate(a), dtype=torch.float32, device=self.device)
        self.load, self.pv, self.p_imp, self.p_exp = f(loads), f(pvs), f(pis), f(pes)
        self.owner = torch.as_tensor(np.concatenate(owner), device=self.device)
        self.split = torch.as_tensor(np.concatenate(split), device=self.device)

        G = len(self.grid_ids)
        self.params = {}
        for key in PARAM_KEYS:
            vals = [finite(getattr(self.configs[g], key), BIG) for g in self.grid_ids]
            self.params[key] = torch.tensor(vals, dtype=torch.float32, device=self.device)
        self.descriptor = torch.tensor(
            np.stack([grid_descriptor(self.configs[g], self.meta[g]) for g in self.grid_ids]),
            dtype=torch.float32, device=self.device)
        self.cost_scale = torch.tensor(
            [self.configs[g].e_max * self.meta[g]["mean_import_price"] for g in self.grid_ids],
            dtype=torch.float32, device=self.device)
        self.n_grids = G

    def day_indices(self, grid_local, split):
        """Global day indices of one grid (local index) in a split."""
        m = (self.owner == grid_local)
        if split is not None:
            m &= (self.split == SPLITS[split])
        return torch.nonzero(m).squeeze(1)


class BatchedMicrogridEnv:
    def __init__(self, bank, n_envs, split="train", grid_weights=None,
                 noise_level=0.0, variable_noise=False, correlation=0.9,
                 price_noise_ratio=0.0, legacy_obs=False, scale_reward=True, seed=0,
                 legacy_soc_limits=False):
        self.bank = bank
        self.legacy_soc_limits = legacy_soc_limits
        self.device = bank.device
        self.N = n_envs
        self.split = split
        self.noise_level = noise_level
        self.variable_noise = variable_noise
        self.correlation = correlation
        self.price_noise_ratio = price_noise_ratio
        self.legacy_obs = legacy_obs
        self.scale_reward = scale_reward
        self.obs_dim = LEGACY_OBS_DIM if legacy_obs else OBS_DIM
        self.gen = torch.Generator(device=self.device)
        self.gen.manual_seed(seed)

        # fixed env -> grid assignment (proportional to weights)
        G = bank.n_grids
        w = np.ones(G) if grid_weights is None else np.asarray(grid_weights, float)
        counts = np.floor(w / w.sum() * n_envs).astype(int)
        counts[: n_envs - counts.sum()] += 1
        self.env_grid = torch.as_tensor(np.repeat(np.arange(G), counts), device=self.device)

        # per-grid pools of day indices for sampling
        pools = [bank.day_indices(k, split) for k in range(G)]
        L = max(len(p) for p in pools)
        self.pool = torch.zeros(G, L, dtype=torch.long, device=self.device)
        self.pool_n = torch.zeros(G, dtype=torch.long, device=self.device)
        for k, p in enumerate(pools):
            self.pool[k, : len(p)] = p
            self.pool_n[k] = len(p)

        hours = torch.arange(T, device=self.device, dtype=torch.float32)
        self.sin_h = torch.sin(2 * math.pi * hours / 24.0)
        self.cos_h = torch.cos(2 * math.pi * hours / 24.0)
        self.reset()

    # ------------------------------------------------------------ helpers --
    def _rand(self, *shape):
        return torch.rand(*shape, generator=self.gen, device=self.device)

    def _randn(self, *shape):
        return torch.randn(*shape, generator=self.gen, device=self.device)

    def _ar1(self, sigma):
        """Correlated AR(1) relative noise [N, T] (ScenarioGenerator.add_noise)."""
        rho = self.correlation
        eps = self._randn(self.N, T)
        noise = torch.empty(self.N, T, device=self.device)
        noise[:, 0] = eps[:, 0] * sigma
        innov = sigma * math.sqrt(max(1 - rho ** 2, 0.0))
        for t in range(1, T):
            noise[:, t] = rho * noise[:, t - 1] + eps[:, t] * innov
        return noise

    @staticmethod
    def _perturb(profile, noise, clip_min):
        clip_max = profile.max(dim=1, keepdim=True).values * 2
        return torch.minimum(torch.clamp(profile * (1 + noise), min=clip_min), clip_max.clamp(min=clip_min))

    # -------------------------------------------------------------- reset --
    def reset(self, day_idx=None, init_soc=None, noise_level=None, actual=None):
        """Start a new day in every env.

        day_idx    : LongTensor [N] global day indices (default: random from split)
        init_soc   : Tensor [N] initial SOC (default: U(0.2, 0.8))
        noise_level: override forecast error level (scalar or [N])
        actual     : optional dict with explicit actual 'load'/'pv'/'price_import' [N,T]
        """
        N, dev = self.N, self.device
        if day_idx is None:
            r = (self._rand(N) * self.pool_n[self.env_grid]).long()
            day_idx = self.pool[self.env_grid, r]
        else:
            day_idx = torch.as_tensor(day_idx, device=dev, dtype=torch.long)
            self.env_grid = self.bank.owner[day_idx]
        self.day_idx = day_idx
        g = self.env_grid
        self.p = {k: v[g] for k, v in self.bank.params.items()}
        self.desc = self.bank.descriptor[g]
        self.scale = self.bank.cost_scale[g] if self.scale_reward else torch.ones(N, device=dev)

        # forecasts (what the agent observes)
        self.f_load = self.bank.load[day_idx]
        self.f_pv = self.bank.pv[day_idx]
        self.f_pimp = self.bank.p_imp[day_idx]
        self.f_pexp = self.bank.p_exp[day_idx]

        # actuals (what the physics uses)
        lvl = self.noise_level if noise_level is None else noise_level
        lvl = torch.as_tensor(lvl, dtype=torch.float32, device=dev).expand(N).clone()
        if self.variable_noise and noise_level is None:
            lvl = self._rand(N) * self.noise_level
        self.current_noise = lvl
        if actual is not None:
            self.a_load, self.a_pv = actual["load"], actual["pv"]
            self.a_pimp = actual.get("price_import", self.f_pimp)
            self.a_pexp = actual.get("price_export", self.f_pexp)
        elif float(lvl.max()) > 0:
            s = lvl[:, None]
            self.a_pv = self._perturb(self.f_pv, self._ar1(1.0) * s, 0.0)
            self.a_load = self._perturb(self.f_load, self._ar1(1.0) * s, 0.1)
            if self.price_noise_ratio > 0:
                pn = self._ar1(1.0) * s * self.price_noise_ratio
                self.a_pimp = self._perturb(self.f_pimp, pn, 0.01)
                self.a_pexp = self.f_pexp * (self.a_pimp / self.f_pimp.clamp(min=1e-6))
            else:
                self.a_pimp, self.a_pexp = self.f_pimp, self.f_pexp
        else:
            self.a_load, self.a_pv, self.a_pimp, self.a_pexp = self.f_load, self.f_pv, self.f_pimp, self.f_pexp

        f_net = self.f_load - self.f_pv
        self.max_net = torch.clamp(f_net.abs().max(dim=1).values, min=1.0)
        self.max_price = torch.clamp(self.f_pimp.max(dim=1).values, min=0.01)

        # state
        self.t = 0
        if init_soc is None:
            init_soc = 0.2 + 0.6 * self._rand(N)
        self.soc = torch.clamp(torch.as_tensor(init_soc, dtype=torch.float32, device=dev).expand(N).clone(),
                               min=self.p["e_min_ratio"], max=torch.ones(N, device=dev))
        self.init_soc = self.soc.clone()
        self.prev_bp = torch.zeros(N, device=dev)
        self.last_err = torch.zeros(N, device=dev)
        self.hist = torch.zeros(N, HIST, device=dev)
        self.run_max_imp = torch.zeros(N, device=dev)
        z = lambda: torch.zeros(N, device=dev)
        self.tot = {k: z() for k in ("cost", "energy", "peak", "demand", "unserved_cost", "unserved_kwh",
                                     "curtailed_kwh", "degradation", "terminal", "throughput",
                                     "peak_violations", "import_kwh", "export_kwh")}
        self.traj = {k: torch.zeros(N, T, device=dev) for k in ("soc", "bat", "grid", "cost")}
        return self._obs()

    # ---------------------------------------------------------------- obs --
    def _obs(self):
        N, h = self.N, self.t
        mn, mp = self.max_net[:, None], self.max_price[:, None]
        f_net = self.f_load - self.f_pv
        if h < T:
            net_now = f_net[:, h]
            price_now = self.f_pimp[:, h]
            sin_h, cos_h = self.sin_h[h].expand(N), self.cos_h[h].expand(N)
        else:   # terminal observation, as MicrogridEnv (profiles read as 0)
            net_now = torch.zeros(N, device=self.device)
            price_now = torch.zeros(N, device=self.device)
            sin_h = torch.full((N,), math.sin(2 * math.pi), device=self.device)
            cos_h = torch.ones(N, device=self.device)
        idx = (torch.arange(1, T + 1, device=self.device) + h) % 24
        fc = torch.stack([
            torch.clamp(f_net[:, idx] / mn, -1, 1),
            self.f_pimp[:, idx] / mp - 0.5,
            self.f_pv[:, idx] / mn,
            self.sin_h[idx].expand(N, T),
            self.cos_h[idx].expand(N, T),
        ], dim=2).reshape(N, 5 * T)
        base = torch.stack([
            (self.soc - 0.5) * 2,
            torch.clamp(net_now / self.max_net, -1, 1),
            price_now / self.max_price - 0.5,
            sin_h, cos_h,
            self.prev_bp / self.p["p_bat_max"],
            torch.clamp(self.last_err / self.max_net, -1, 1),
        ], dim=1)
        parts = [base, self.hist, fc]
        if not self.legacy_obs:
            extra = torch.stack([
                torch.clamp(self.max_net / self.p["p_bat_max"], max=10.0) / 5.0,
                self.max_price / 0.3,
                torch.clamp(self.run_max_imp / self.max_net, max=2.0),
            ], dim=1)
            parts += [self.desc, extra]
        return torch.cat(parts, dim=1)

    # --------------------------------------------------------------- step --
    def step(self, action):
        p, h = self.p, self.t
        a = torch.as_tensor(action, dtype=torch.float32, device=self.device).reshape(self.N)
        req = torch.clamp(a, -1, 1) * p["p_bat_max"]

        # ramp limit
        change = req - self.prev_bp
        change = torch.where(change.abs() > p["ramp_rate"], torch.sign(change) * p["ramp_rate"], change)
        bp = self.prev_bp + change

        load, pv = self.a_load[:, h], self.a_pv[:, h]
        p_imp, p_exp = self.a_pimp[:, h], self.a_pexp[:, h]
        net = load - pv

        # SOC window. Legacy (MicrogridEnv) limits ignore efficiency, so the
        # SOC clip silently loses charge near full and *creates* up to 5% free
        # energy near empty. The default uses the exact limits, which are the
        # same feasible set as the LP: E + eta_c*Pc <= Emax, E - Pd/eta_d >= Emin.
        max_ch = (1.0 - self.soc) * p["e_max"]
        avail = (self.soc - p["e_min_ratio"]) * p["e_max"]
        if not self.legacy_soc_limits:
            max_ch = max_ch / p["eta_charge"]
            avail = avail * p["eta_discharge"]
        bp = torch.where(bp > 0, torch.minimum(torch.minimum(bp, max_ch), p["p_bat_max"]),
                         torch.maximum(torch.maximum(bp, -avail), -p["p_bat_max"]))

        # grid connection limits: curb battery first
        grid = net + bp
        over_imp = torch.clamp(grid - p["import_cap_kw"], min=0)
        bp = bp - torch.where(bp > 0, torch.minimum(bp, over_imp), torch.zeros_like(bp))
        grid = net + bp
        over_exp = torch.clamp(-p["export_cap_kw"] - grid, min=0)
        bp = bp + torch.where(bp < 0, torch.minimum(-bp, over_exp), torch.zeros_like(bp))

        e2b = torch.where(bp > 0, bp * p["eta_charge"], bp / p["eta_discharge"])
        new_soc = torch.clamp(self.soc + e2b / p["e_max"], min=p["e_min_ratio"], max=torch.ones_like(bp))
        dE = (new_soc - self.soc) * p["e_max"]

        grid = net + bp
        unserved = torch.clamp(grid - p["import_cap_kw"], min=0)
        grid = grid - unserved
        curtailed = torch.clamp(-p["export_cap_kw"] - grid, min=0)
        grid = grid + curtailed

        energy = torch.where(grid >= 0, grid * p_imp, grid * p_exp)
        peak_v = torch.clamp(grid - p["p_grid_peak"], min=0)
        peak = peak_v * p["peak_penalty_rate"]
        imp = torch.clamp(grid, min=0)
        demand = p["demand_charge_per_kw_day"] * torch.clamp(imp - self.run_max_imp, min=0)
        self.run_max_imp = torch.maximum(self.run_max_imp, imp)
        unserved_cost = unserved * p["voll"]
        thr = dE.abs()
        deg = thr * p["degradation_cost_per_kwh"]
        cost = energy + peak + demand + unserved_cost + deg

        # state update
        self.soc = new_soc
        self.prev_bp = bp
        f_net = self.f_load[:, h] - self.f_pv[:, h]
        self.last_err = net - f_net
        self.hist = torch.roll(self.hist, -1, dims=1)
        self.hist[:, -1] = torch.clamp(net / self.max_net, -1, 1)

        tot = self.tot
        tot["energy"] += energy; tot["peak"] += peak; tot["demand"] += demand
        tot["unserved_cost"] += unserved_cost; tot["unserved_kwh"] += unserved
        tot["curtailed_kwh"] += curtailed; tot["degradation"] += deg
        tot["throughput"] += thr; tot["peak_violations"] += (peak_v > 0).float()
        tot["import_kwh"] += imp; tot["export_kwh"] += torch.clamp(-grid, min=0)
        tot["cost"] += cost
        self.traj["soc"][:, h] = new_soc; self.traj["bat"][:, h] = bp
        self.traj["grid"][:, h] = grid; self.traj["cost"][:, h] = cost

        self.t += 1
        done = self.t >= T
        if done:
            dev_kwh = (self.soc - p["target_final_soc"]).abs() * p["e_max"]
            term = dev_kwh * p["terminal_penalty_per_kwh"]
            tot["terminal"] = term
            tot["cost"] += term
            cost = cost + term
        reward = -cost / self.scale
        return self._obs(), reward, done, {"cost": cost}

    # ------------------------------------------------------------ results --
    def summary(self):
        """Per-env episode totals as numpy arrays (call after the 24th step)."""
        out = {k: v.detach().cpu().numpy() for k, v in self.tot.items()}
        out["final_soc"] = self.soc.cpu().numpy()
        out["init_soc"] = self.init_soc.cpu().numpy()
        out["max_import_kw"] = self.run_max_imp.cpu().numpy()
        out["day_idx"] = self.day_idx.cpu().numpy()
        out["grid"] = np.array([self.bank.grid_ids[i] for i in self.env_grid.cpu().numpy()])
        out["noise"] = self.current_noise.cpu().numpy()
        return out
