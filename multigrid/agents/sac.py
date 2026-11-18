# multigrid/agents/sac.py
# Vectorised SAC for the batched microgrid env, replay buffer on the GPU.
#
# Each vector step adds N transitions; `updates_per_step` gradient steps follow.
# Hour 24 is a true terminal (no bootstrap).

import copy
import math
import time
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F

from multigrid.agents.common import mlp, RunningNorm, Policy, val_score
from multigrid.torch_env import T


@dataclass
class SACConfig:
    n_envs: int = 128
    total_steps: int = 3_000_000
    buffer_size: int = 500_000
    batch_size: int = 2048
    updates_per_step: int = 8
    learning_starts: int = 20_000
    lr: float = 3e-4
    gamma: float = 1.0
    tau: float = 0.005
    hidden: tuple = (256, 256)
    target_entropy: float = -1.0
    norm_warmup_steps: int = 200_000     # obs normaliser frozen after this
    cuda_graph: bool = True              # capture the update block in a CUDA graph


LOG_STD_MIN, LOG_STD_MAX = -5.0, 2.0
HALF_LOG_2PI = 0.5 * math.log(2 * math.pi)


class Actor(nn.Module):
    def __init__(self, obs_dim, hidden):
        super().__init__()
        self.body = mlp(obs_dim, 2, hidden, act=nn.ReLU, out_gain=0.01)

    def forward(self, o, deterministic=False):
        mu, log_std = self.body(o).unbind(-1)
        if deterministic:
            return torch.tanh(mu), None
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = log_std.exp()
        u = mu + std * torch.randn_like(mu)
        a = torch.tanh(u)
        logp = (-0.5 * ((u - mu) / std) ** 2 - log_std - HALF_LOG_2PI
                - torch.log(1 - a ** 2 + 1e-6))
        return a, logp


class Critic(nn.Module):
    def __init__(self, obs_dim, hidden):
        super().__init__()
        self.q1 = mlp(obs_dim + 1, 1, hidden, act=nn.ReLU)
        self.q2 = mlp(obs_dim + 1, 1, hidden, act=nn.ReLU)

    def forward(self, o, a):
        x = torch.cat([o, a[:, None]], dim=1)
        return self.q1(x).squeeze(-1), self.q2(x).squeeze(-1)


class SACAgent(Policy):
    name = "SAC"

    def __init__(self, obs_dim, cfg=None, device="cuda"):
        self.cfg = cfg or SACConfig()
        self.device = torch.device(device)
        c = self.cfg
        self.obs_dim = obs_dim
        self.norm = RunningNorm(obs_dim).to(self.device)
        self.actor = Actor(obs_dim, c.hidden).to(self.device)
        self.critic = Critic(obs_dim, c.hidden).to(self.device)
        self.critic_t = Critic(obs_dim, c.hidden).to(self.device)
        self.critic_t.load_state_dict(self.critic.state_dict())
        self.log_alpha = torch.zeros(1, device=self.device, requires_grad=True)
        cuda = self.device.type == "cuda"
        kw = {"fused": True, "capturable": True} if cuda else {}
        self.opt_a = torch.optim.Adam(self.actor.parameters(), lr=c.lr, **kw)
        self.opt_c = torch.optim.Adam(self.critic.parameters(), lr=c.lr, **kw)
        self.opt_al = torch.optim.Adam([self.log_alpha], lr=c.lr, **({"capturable": True} if cuda else {}))
        self.graph = None
        self._p = list(self.critic.parameters())
        self._pt = list(self.critic_t.parameters())
        self.history = []

    @torch.no_grad()
    def act(self, obs, deterministic=True):
        a, _ = self.actor(self.norm(obs), deterministic=deterministic)
        return a

    def _update(self, o, a, r, o2, d):
        c = self.cfg
        alpha = self.log_alpha.exp().detach()
        with torch.no_grad():
            a2, logp2 = self.actor(o2)
            q1t, q2t = self.critic_t(o2, a2)
            target = r + c.gamma * (1 - d) * (torch.min(q1t, q2t) - alpha * logp2)
        q1, q2 = self.critic(o, a)
        lc = F.mse_loss(q1, target) + F.mse_loss(q2, target)
        self.opt_c.zero_grad(set_to_none=False); lc.backward(); self.opt_c.step()

        an, logp = self.actor(o)
        q1n, q2n = self.critic(o, an)
        la = (alpha * logp - torch.min(q1n, q2n)).mean()
        self.opt_a.zero_grad(set_to_none=False); la.backward(); self.opt_a.step()

        lal = -(self.log_alpha * (logp.detach() + c.target_entropy)).mean()
        self.opt_al.zero_grad(set_to_none=False); lal.backward(); self.opt_al.step()

        with torch.no_grad():
            torch._foreach_lerp_(self._pt, self._p, c.tau)

    def _update_block(self):
        """`updates_per_step` gradient steps on minibatches from the replay buffer."""
        c = self.cfg
        for _ in range(c.updates_per_step):
            j = (torch.rand(c.batch_size, device=self.device) * self._size).long()
            self._update(self.norm(self.B_o[j]), self.B_a[j], self.B_r[j],
                         self.norm(self.B_o2[j]), self.B_d[j])

    def _run_updates(self):
        if not (self.cfg.cuda_graph and self.device.type == "cuda"):
            return self._update_block()
        if self.graph is None:
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):                  # warm-up (real updates)
                for _ in range(3):
                    self._update_block()
            torch.cuda.current_stream().wait_stream(s)
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph):
                self._update_block()
        self.graph.replay()

    def train(self, env, log_every=50, val_fn=None, verbose=True):
        c, dev, N, D = self.cfg, self.device, env.N, env.obs_dim
        cap = c.buffer_size
        # raw (unnormalised) obs are stored; normalised on sampling
        B_o = torch.zeros(cap, D, device=dev); B_o2 = torch.zeros(cap, D, device=dev)
        B_a = torch.zeros(cap, device=dev); B_r = torch.zeros(cap, device=dev); B_d = torch.zeros(cap, device=dev)
        self.B_o, self.B_o2, self.B_a, self.B_r, self.B_d = B_o, B_o2, B_a, B_r, B_d
        self._size = torch.zeros(1, device=dev)
        self.graph = None
        ptr, size, steps, ep = 0, 0, 0, 0
        t0 = time.time()
        best = (float("inf"), None, 0)
        n_vec_steps = max(1, c.total_steps // N)
        obs = env.reset()
        for vs in range(n_vec_steps):
            if steps < c.norm_warmup_steps:
                self.norm.update(obs)
            if steps < c.learning_starts:
                a = torch.rand(N, device=dev) * 2 - 1
            else:
                with torch.no_grad():
                    a, _ = self.actor(self.norm(obs))
            obs2, r, done, _ = env.step(a)
            idx = (torch.arange(N, device=dev) + ptr) % cap
            B_o[idx], B_a[idx], B_r[idx], B_o2[idx] = obs, a, r, obs2
            B_d[idx] = 1.0 if done else 0.0
            ptr = (ptr + N) % cap; size = min(size + N, cap); steps += N
            if done:
                ep_cost = env.tot["cost"].mean().item()
                obs = env.reset(); ep += 1
            else:
                obs = obs2

            if steps >= c.learning_starts:
                self._size.fill_(size)
                self._run_updates()

            if done and (ep % log_every == 0 or vs == n_vec_steps - 1):
                rec = {"episode": ep, "steps": steps, "train_cost": ep_cost,
                       "alpha": self.log_alpha.exp().item(), "time": time.time() - t0}
                if val_fn is not None:
                    rec["val"] = val_fn(self)
                    sc = val_score(rec["val"])
                    if sc < best[0]:
                        best = (sc, (copy.deepcopy(self.actor.state_dict()),
                                     copy.deepcopy(self.norm.state_dict())), steps)
                self.history.append(rec)
                if verbose:
                    msg = "  ep %5d  steps %9d  cost %9.3f  alpha %.4f  %5.0fs" % (
                        ep, steps, ep_cost, rec["alpha"], rec["time"])
                    if "val" in rec:
                        msg += "  val %s" % rec["val"]
                    print(msg, flush=True)
        self.graph = None
        self.B_o = self.B_o2 = self.B_a = self.B_r = self.B_d = None   # free the buffer
        if best[1] is not None:                       # keep best-on-validation weights
            self.actor.load_state_dict(best[1][0]); self.norm.load_state_dict(best[1][1])
            self.best_step = best[2]
            if verbose:
                print("  restored best validation checkpoint (step %d, score %.4f)" % (best[2], best[0]))
        return self.history

    def state(self):
        return {"algo": "sac", "obs_dim": self.obs_dim, "cfg": asdict(self.cfg),
                "norm": self.norm.state_dict(), "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(), "history": self.history}

    @classmethod
    def from_state(cls, st, device="cuda"):
        cfg = SACConfig(**{k: tuple(v) if k == "hidden" else v for k, v in st["cfg"].items()})
        a = cls(st["obs_dim"], cfg, device)
        a.norm.load_state_dict(st["norm"]); a.actor.load_state_dict(st["actor"])
        a.critic.load_state_dict(st["critic"])
        a.history = st.get("history", [])
        return a
