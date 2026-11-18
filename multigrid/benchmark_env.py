# multigrid/benchmark_env.py
# Env throughput (env-steps/s) for several batch sizes on CPU and CUDA.
import time
import torch
from multigrid.grids import GRID_IDS
from multigrid.torch_env import DayBank, BatchedMicrogridEnv, T


def bench(device, n_envs, episodes=20, noise=0.1):
    bank = DayBank(GRID_IDS, device=device)
    env = BatchedMicrogridEnv(bank, n_envs, noise_level=noise, variable_noise=True)
    a = torch.zeros(n_envs, device=device)
    for _ in range(2):  # warm-up
        env.reset()
        for t in range(T):
            env.step(a)
    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(episodes):
        env.reset()
        for t in range(T):
            env.step(torch.rand(n_envs, device=device) * 2 - 1)
    if device == "cuda":
        torch.cuda.synchronize()
    return episodes * T * n_envs / (time.time() - t0)


if __name__ == "__main__":
    devs = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
    for dev in devs:
        for n in (256, 1024, 4096, 16384):
            print("%-5s N=%-6d %10.0f env-steps/s" % (dev, n, bench(dev, n)))
