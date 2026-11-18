# multigrid/agents/ppo.py
# PPO for the batched microgrid env, fully on-GPU.
#
# One iteration = one synchronised 24-step day in every env (N*24 samples).
# Episodes end at hour 24 (true terminal), so there is no bootstrapping.
# Actions are tanh-squashed Gaussians (a = tanh(u)); the PPO ratio only needs
# log N(u) because the tanh Jacobian cancels for a fixed sample u.

import time
import math
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn

import copy

from multigrid.agents.common import mlp, RunningNorm, Policy, val_score
from multigrid.torch_env import T


@dataclass
class PPOConfig:
    n_envs: int = 4096
    total_steps: int = 20_000_000
    lr: float = 3e-4
    gamma: float = 1.0           # finite 24 h horizon: undiscounted daily cost
    lam: float = 0.95
    clip: float = 0.2
    epochs: int = 5
    minibatch: int = 8192
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    hidden: tuple = (256, 256)
    init_log_std: float = -0.5
    anneal_lr: bool = True


class ActorCritic(nn.Module):
    def __init__(self, obs_dim, hidden=(256, 256), init_log_std=-0.5):
        super().__init__()
        self.norm = RunningNorm(obs_dim)
        self.pi = mlp(obs_dim, 1, hidden, out_gain=0.01)
        self.v = mlp(obs_dim, 1, hidden, out_gain=1.0)
        self.log_std = nn.Parameter(torch.full((1,), init_log_std))

    def dist(self, obs_n):
        mu = self.pi(obs_n).squeeze(-1)
        return torch.distributions.Normal(mu, self.log_std.exp().expand_as(mu))

    def value(self, obs_n):
        return self.v(obs_n).squeeze(-1)


class PPOAgent(Policy):
    name = "PPO"

    def __init__(self, obs_dim, cfg=None, device="cuda"):
        self.cfg = cfg or PPOConfig()
        self.device = torch.device(device)
        self.net = ActorCritic(obs_dim, self.cfg.hidden, self.cfg.init_log_std).to(self.device)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=self.cfg.lr, eps=1e-5)
        self.obs_dim = obs_dim
        self.history = []

    @torch.no_grad()
    def act(self, obs, deterministic=True):
        o = self.net.norm(obs)
        if deterministic:
            return torch.tanh(self.net.pi(o).squeeze(-1))
        return torch.tanh(self.net.dist(o).sample())

    def train(self, env, log_every=10, val_fn=None, verbose=True):
        cfg, dev, N = self.cfg, self.device, env.N
        n_iter = max(1, cfg.total_steps // (N * T))
        D = env.obs_dim
        buf_obs = torch.zeros(T, N, D, device=dev)
        buf_act = torch.zeros(T, N, device=dev)
        buf_logp = torch.zeros(T, N, device=dev)
        buf_rew = torch.zeros(T, N, device=dev)
        buf_val = torch.zeros(T + 1, N, device=dev)
        t0 = time.time()
        best = (float("inf"), None, 0)

        for it in range(n_iter):
            if cfg.anneal_lr:
                for gp in self.opt.param_groups:
                    gp["lr"] = cfg.lr * (1 - it / n_iter)

            # ---- rollout: one day in every env
            obs = env.reset()
            with torch.no_grad():
                for t in range(T):
                    self.net.norm.update(obs)
                    o = self.net.norm(obs)
                    d = self.net.dist(o)
                    u = d.sample()                     # pre-squash sample
                    buf_obs[t], buf_act[t], buf_logp[t] = obs, u, d.log_prob(u)
                    buf_val[t] = self.net.value(o)
                    obs, r, done, _ = env.step(torch.tanh(u))
                    buf_rew[t] = r
                buf_val[T] = 0.0                       # terminal

                # ---- GAE
                adv = torch.zeros(T, N, device=dev)
                last = torch.zeros(N, device=dev)
                for t in reversed(range(T)):
                    delta = buf_rew[t] + cfg.gamma * buf_val[t + 1] - buf_val[t]
                    last = delta + cfg.gamma * cfg.lam * last
                    adv[t] = last
                ret = adv + buf_val[:T]

            # ---- update
            b_obs = self.net.norm(buf_obs.reshape(T * N, D))
            b_act, b_logp = buf_act.reshape(-1), buf_logp.reshape(-1)
            b_adv, b_ret = adv.reshape(-1), ret.reshape(-1)
            B = T * N
            mb = min(cfg.minibatch, B)
            for _ in range(cfg.epochs):
                perm = torch.randperm(B, device=dev)
                for s in range(0, B, mb):
                    idx = perm[s:s + mb]
                    d = self.net.dist(b_obs[idx])
                    logp = d.log_prob(b_act[idx])
                    ratio = (logp - b_logp[idx]).exp()
                    a_ = b_adv[idx]
                    a_ = (a_ - a_.mean()) / (a_.std() + 1e-8)
                    pg = -torch.min(ratio * a_, torch.clamp(ratio, 1 - cfg.clip, 1 + cfg.clip) * a_).mean()
                    v = self.net.value(b_obs[idx])
                    vl = 0.5 * ((v - b_ret[idx]) ** 2).mean()
                    ent = d.entropy().mean()
                    loss = pg + cfg.vf_coef * vl - cfg.ent_coef * ent
                    self.opt.zero_grad(set_to_none=True)
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.net.parameters(), cfg.max_grad_norm)
                    self.opt.step()

            ep_cost = env.tot["cost"].mean().item()
            ep_ret = buf_rew.sum(0).mean().item()
            rec = {"iter": it, "steps": (it + 1) * N * T, "train_cost": ep_cost,
                   "train_return": ep_ret, "std": self.net.log_std.exp().item(),
                   "time": time.time() - t0}
            if val_fn is not None and (it % log_every == 0 or it == n_iter - 1):
                rec["val"] = val_fn(self)
                sc = val_score(rec["val"])
                if sc < best[0]:
                    best = (sc, copy.deepcopy(self.net.state_dict()), rec["steps"])
            self.history.append(rec)
            if verbose and (it % log_every == 0 or it == n_iter - 1):
                msg = "  it %4d/%d  steps %9d  return %8.3f  std %.3f  %5.0fs" % (
                    it, n_iter, rec["steps"], ep_ret, rec["std"], rec["time"])
                if "val" in rec:
                    msg += "  val %s" % rec["val"]
                print(msg, flush=True)
        if best[1] is not None:                       # keep best-on-validation weights
            self.net.load_state_dict(best[1])
            self.best_step = best[2]
            if verbose:
                print("  restored best validation checkpoint (step %d, score %.4f)" % (best[2], best[0]))
        return self.history

    # ---- persistence
    def state(self):
        return {"algo": "ppo", "obs_dim": self.obs_dim, "cfg": asdict(self.cfg),
                "net": self.net.state_dict(), "history": self.history}

    @classmethod
    def from_state(cls, st, device="cuda"):
        cfg = PPOConfig(**{k: tuple(v) if k == "hidden" else v for k, v in st["cfg"].items()})
        a = cls(st["obs_dim"], cfg, device)
        a.net.load_state_dict(st["net"])
        a.history = st.get("history", [])
        return a
