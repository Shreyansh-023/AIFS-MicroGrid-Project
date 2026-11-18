# multigrid/agents/common.py
# Shared building blocks: MLP, running observation normaliser, checkpoint I/O.

import torch
import torch.nn as nn


def mlp(inp, out, hidden=(256, 256), act=nn.Tanh, out_gain=1.0):
    layers, d = [], inp
    for h in hidden:
        lin = nn.Linear(d, h)
        nn.init.orthogonal_(lin.weight, gain=2 ** 0.5)
        nn.init.zeros_(lin.bias)
        layers += [lin, act()]
        d = h
    last = nn.Linear(d, out)
    nn.init.orthogonal_(last.weight, gain=out_gain)
    nn.init.zeros_(last.bias)
    layers.append(last)
    return nn.Sequential(*layers)


class RunningNorm(nn.Module):
    """Running mean/var observation normaliser (parallel Welford update)."""

    def __init__(self, dim, clip=10.0, eps=1e-8):
        super().__init__()
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("var", torch.ones(dim))
        self.register_buffer("count", torch.tensor(1e-4))
        self.clip, self.eps = clip, eps
        self.frozen = False

    @torch.no_grad()
    def update(self, x):
        if self.frozen:
            return
        x = x.reshape(-1, x.shape[-1])
        b_mean, b_var, b_n = x.mean(0), x.var(0, unbiased=False), x.shape[0]
        delta = b_mean - self.mean
        tot = self.count + b_n
        self.mean += delta * b_n / tot
        m2 = self.var * self.count + b_var * b_n + delta ** 2 * self.count * b_n / tot
        self.var.copy_(m2 / tot)          # in place: CUDA graphs keep references
        self.count.copy_(tot)

    def forward(self, x):
        # Features that never varied during training (e.g. the forecast-error
        # input when training without noise, or the grid descriptor of a
        # single-grid agent) are only centred. Dividing by their ~0 std would
        # blow any unseen variation up by ~1e4 and saturate the clip. Inputs
        # seen in training are unaffected (x == mean -> 0 either way).
        std = torch.sqrt(self.var + self.eps)
        std = torch.where(self.var < 1e-6, torch.ones_like(std), std)
        return torch.clamp((x - self.mean) / std, -self.clip, self.clip)


class Policy:
    """Uniform interface used by evaluation: act(obs[N,D]) -> action[N] in [-1, 1]."""

    name = "policy"

    def act(self, obs):
        raise NotImplementedError


def val_score(v):
    """Scalar used for best-checkpoint selection (lower is better)."""
    if isinstance(v, dict):
        return float(v["_score"]) if "_score" in v else float(sum(x for x in v.values()))
    return float(v)
