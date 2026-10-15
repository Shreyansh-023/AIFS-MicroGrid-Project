# AIFS Microgrid — RL Battery Dispatch

Reinforcement learning for battery energy management in a residential microgrid,
with an interactive dashboard for exploring the results.

Three RL agents (SAC, PPO, and a domain-randomised "Robust SAC") control a
13.5 kWh battery over a 24-hour horizon, trading against time-of-use electricity
prices. They are benchmarked against a linear program given perfect foresight —
then both sides have their forecasts corrupted to see which strategy survives.

**The finding:** with a clean forecast the LP is unbeatable, since it solves the
whole day exactly. As forecast error grows, the LP degrades sharply — it commits
to a schedule before the day starts and cannot revise it — while the RL agents
re-decide every hour and hold their ground.

## Quick start

```bash
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt

python main.py --demo          # ~1 min smoke test
python main.py                 # full pipeline (~2 h with CUDA)
```

## The dashboard

An interactive page over the simulation output: hour-by-hour episode replay,
agent-vs-agent comparison, and a forecast-error slider that makes the robustness
result tangible.

```bash
python export_web_data.py      # results -> web/public/data/simulation.json
cd web && npm install && npm run dev
```

It runs without trained models too — the export falls back to the LP benchmark
and baseline agents, and the robustness section shows a placeholder until models
exist. See [`web/README.md`](web/README.md) for details.

## How it works

**Environment** (`microgrid_env.py`) — Gymnasium env with a 131-dimensional
observation: current state (SOC, net power, price, hour encoding), a short
history buffer, and a 24-hour forecast window that makes up over 90% of the
input. One continuous action per hour: battery power in ±5 kW.

**Cost function** — grid energy at TOU prices (exports earn 40% of retail), a
demand charge above 10 kW, battery degradation per kWh cycled, and a terminal
penalty that forces the agent to hand the pack back at 50% charge rather than
draining it for a one-day payday.

**Forecast vs actual** — the env keeps the two separate. The agent observes
forecasts; the physics runs on actuals. Robustness testing perturbs the actuals
with correlated AR(1) noise (ρ = 0.9), so an hour that forecasts badly tends to
stay bad — which is what makes a pre-committed schedule risky.

**Agents** (`rl_agents.py`) — SAC and PPO from Stable-Baselines3, plus R-SAC:
the same SAC trained on an environment that redraws its noise level each episode,
so the policy learns to distrust its own forecast. Rule-based and no-op agents
provide floors.

**Benchmark** (`lp_solver.py`) — SciPy HiGHS linear program over charge,
discharge, peak-violation and grid import/export variables, subject to power
balance, SOC bounds, ramp limits and a terminal SOC equality.

## Layout

```
main.py               pipeline: train -> evaluate -> visualise -> stress test
microgrid_env.py      Gymnasium environment
rl_agents.py          SAC / PPO / R-SAC factories and baselines
lp_solver.py          perfect-foresight LP benchmark
evaluation.py         multi-episode evaluation and optimality gap
robustness_test.py    forecast-error sweep
visualization.py      publication plots -> results/*.pdf
export_web_data.py    results -> JSON for the dashboard
web/                  React + Recharts dashboard
```

## Data

Supports Pecan Street Dataport and generic consumption/production CSVs, placed in
the project root. Data files are not committed. With no data present the system
generates synthetic solar and load profiles, so everything runs out of the box.

## Notes on performance

Training is bound by kernel-launch latency, not GPU compute: SAC fires many small
sequential operations that cannot saturate a laptop GPU. Measured on an RTX 3050,
CUDA gives roughly 2.25× over CPU (~2 h vs ~4.5 h for all three agents). Larger
batches do not help — a batch of 1024 costs the same as 256.

<img width="1109" height="480" alt="image" src="https://github.com/user-attachments/assets/1214c9e3-fb95-4f5d-836d-7ef23b24c40c" />
