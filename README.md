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

The project then grows from one home to **six microgrid archetypes** with a year of
real 2018 data each (see [Multi-grid study](#multi-grid-study-multigrid-run_multigridpy)).

| | |
|---|---|
| **GitHub** | <https://github.com/Shreyansh-023/AIFS-MicroGrid-Project> |
| **Report (IEEE format)** | [`report/IEEE_Report_Multigrid_RL.pdf`](report/IEEE_Report_Multigrid_RL.pdf) · Word: [`report/IEEE_Report_Multigrid_RL.docx`](report/IEEE_Report_Multigrid_RL.docx) · Overleaf: upload [`report/report_overleaf.zip`](report/report_overleaf.zip) |
| **Dashboard** | `web/` (React), with a grid selector for all six microgrids |

## Team

Department of Artificial Intelligence, Sardar Vallabhbhai National Institute of
Technology (SVNIT), Surat, India.

- Shreyansh Mishra
- Aryan Sawant
- Ankit Yadav
- Chirag Bhut
- Mihika Chawda
- Nitesh Funde

## Implementation environment

The results in this repository were produced on:

| Component | Version / spec |
|---|---|
| OS | Windows 11 Home (64-bit) |
| GPU | NVIDIA GeForce RTX 3050 Laptop, 4 GB, driver 610.78 |
| Python | 3.10 |
| Multi-grid study | PyTorch 2.6.0 + CUDA 12.4, NumPy 1.26, SciPy 1.15 (HiGHS), pandas 2.3, pyarrow 21, Matplotlib 3.10 |
| Single-home study | PyTorch 2.5.1 + CUDA 12.1, Gymnasium 0.29.1, Stable-Baselines3 2.1.0 |
| Dashboard | Node.js 22, React 19, Vite 8, Recharts 3, TypeScript 6 |
| Report | LaTeX (IEEEtran, MiKTeX), also builds on Overleaf |

CUDA is optional: everything runs on CPU, only more slowly.

### Setup

```bash
git clone https://github.com/Shreyansh-023/AIFS-MicroGrid-Project.git
cd AIFS-MicroGrid-Project

python -m venv venv
venv\Scripts\activate                 # Windows (Linux/macOS: source venv/bin/activate)

# PyTorch with CUDA (pick the wheel for your driver at https://pytorch.org)
pip install torch --index-url https://download.pytorch.org/whl/cu124

pip install -r requirements.txt            # single-home study (Gymnasium, SB3)
pip install -r requirements-multigrid.txt  # multi-grid study
```

Dashboard: install Node.js 18 or newer, then run `cd web && npm install`.

## Datasets

All data are public. The multi-grid datasets download automatically on first run
(`python -m multigrid.data.build_dataset`, about 5 minutes), need no API keys, and
are cached in `data/raw/`, which is not committed.

| Dataset | Used for | Source |
|---|---|---|
| NREL End-Use Load Profiles, ResStock AMY2018 release 1 | Residential loads (G1, G2, G4, G5) | [OEDI data lake](https://data.openei.org/s3_viewer?bucket=oedi-data-lake&prefix=nrel-pds-building-stock%2Fend-use-load-profiles-for-us-building-stock%2F2021%2F), `s3://oedi-data-lake/nrel-pds-building-stock/end-use-load-profiles-for-us-building-stock/2021/resstock_amy2018_release_1` |
| NREL End-Use Load Profiles, ComStock AMY2018 release 1 | Office (G3) and warehouse (G6) loads | same bucket, `comstock_amy2018_release_1` |
| NREL county weather, AMY2018 | PV generation via a PVWatts-style model | same bucket, `resstock_amy2018_release_1/weather/amy2018/` |
| NYISO day-ahead market zonal LBMP, 2018 (N.Y.C. zone) | Real-time price tariff (G5) | <http://mis.nyiso.com/public/csv/damlbmp/> |
| Pecan Street Dataport (10 homes, 1-min, Aug 2018) | Original single-home study | <https://www.pecanstreet.org/dataport/> (free academic account) |

EV charging sessions for G5 are synthesised from arrival-time and energy
distributions ([`multigrid/data/build_dataset.py`](multigrid/data/build_dataset.py)).
Tariffs are defined in [`multigrid/tariffs.py`](multigrid/tariffs.py).

## Quick start

```bash
python main.py --demo          # ~1 min smoke test (single-home study)
python main.py                 # full single-home pipeline (~2 h with CUDA)

python run_multigrid.py --quick   # multi-grid smoke test (~5 min)
```

## The dashboard

An interactive page over the simulation output: hour-by-hour episode replay,
agent-vs-agent comparison, and a forecast-error slider that makes the robustness
result tangible.

A **grid selector** (top bar, plus the "Six microgrids" section) switches every
view between the original single-home study and the six multi-grid archetypes. The
"Six microgrids" section also has a cross-grid table of how much of the optimal
savings each method captures. The selected grid is kept in the URL
(`?grid=G3`), so a view can be shared.

```bash
python export_web_data.py         # single-home results -> web/public/data/simulation.json
python export_multigrid_web.py    # multi-grid results  -> web/public/data/grids/*.json
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

## Multi-grid study (`multigrid/`, `run_multigrid.py`)

An extension from one home to **six microgrid archetypes**. Each archetype has
its own battery and PV sizing, tariff and grid-connection limits, and a full
year (2018) of real hourly load, PV and price data:

| ID | Archetype | Battery | PV | Tariff / limits | Data |
|---|---|---|---|---|---|
| G1 | Residential, hot (Austin TX) | 13.5 kWh / 5 kW | 6 kWp | TOU, 40 % feed-in, 10 kW soft peak | 3 ResStock homes |
| G2 | Residential, cold (Minneapolis MN) | 10 kWh / 5 kW | 5 kWp | flat $0.14, 25 % feed-in | 3 ResStock homes |
| G3 | Commercial office (Los Angeles CA) | 200 kWh / 60 kW | 120 kWp | commercial TOU + $0.60/kW-day demand charge | ComStock medium office |
| G4 | Rural weak grid (Apache County AZ) | 200 kWh / 40 kW | 80 kWp | import ≤ 30 kW (VoLL $5/kWh), export ≤ 30 kW | 20 ResStock homes |
| G5 | EV community, real-time price (Queens NY) | 60 kWh / 25 kW | 40 kWp | NYISO NYC day-ahead LBMP + $0.10, no export | 10 homes + 8 EVs |
| G6 | Industrial warehouse (Houston TX) | 500 kWh / 125 kW | 300 kWp | industrial TOU + $1.00/kW-day demand charge | ComStock warehouse |

**Data** (downloaded and cached in `data/raw/`; no API keys needed):
- Loads come from the NREL End-Use Load Profiles (ResStock and ComStock, AMY2018).
- PV is computed from NREL's 2018 county weather (GHI, DNI, DHI, temperature) with a PVWatts-style model ([multigrid/data/pv_model.py](multigrid/data/pv_model.py)), so load and PV come from the same 2018 weather.
- Real-time prices are NYISO day-ahead zonal LBMP.
- Each month is split the same way: days 8–14 are test days, days 22–28 validation, and the rest training.

**Fast training.** [multigrid/torch_env.py](multigrid/torch_env.py) steps thousands of microgrids at once as GPU tensors. On an RTX 3050 it runs about 2.3M env-steps/s with 4096 envs, against roughly 80 steps/s for the SB3 setup. PPO and SAC are written directly in PyTorch ([multigrid/agents/](multigrid/agents/)).

**Physics.** The batched env reproduces `MicrogridEnv` exactly, and a test checks this. It adds:
- separate import and export prices;
- a demand charge;
- an import cap, with shed load priced at VoLL;
- an export cap, with curtailment.

It also fixes an efficiency inconsistency in the original SOC limits, which let the battery gain up to 5 % free energy near empty. The old behaviour is kept as `legacy_soc_limits=True`.

**Baselines.**
- A perfect-foresight LP whose objective is exactly the env's daily cost.
- The same LP run open loop on the forecast.
- A shrinking-horizon **MPC**.
- NoOp, the original rule-based agent, and a self-consumption controller.

**Experiments.**

| | Question |
|---|---|
| E1 | Specialists per grid vs LP, MPC and heuristics |
| E2 | One generalist policy for all grids |
| E3 | Leave-one-grid-out zero-shot transfer, plus a cross-grid transfer matrix |
| E4 | Forecast-error sweep (AR(1), ρ = 0.9) per grid |

The headline metric is the **share of the LP's savings captured**, computed on held-out test days and reported as the mean with a 95 % CI over seeds. Wilcoxon tests compare each method with MPC.

```bash
python -m multigrid.data.build_dataset        # download + build all grids (~5 min)
python tests/test_torch_env.py                # env equivalence with MicrogridEnv
python tests/test_lp.py                       # LP regression + consistency
python run_multigrid.py --quick               # smoke test
python run_multigrid.py                       # full study -> results/multigrid/
```

**Training settings**, all tuned on G1/G3 validation days:

| Algorithm | Envs | Steps | Updates | Time per run (RTX 3050) |
|---|---|---|---|---|
| SAC | 128 | 6M | 4 per vector step, batch 1024, captured in a CUDA graph | ~12 min |
| PPO | 512 | 10M | tanh-squashed actions | ~5 min |

Generalists get 1.5× a specialist's budget. Every run keeps its best checkpoint on the *validation* days; test days are only used for reporting.

The "lean" study used for the paper draft takes about 7 h:

```bash
python run_multigrid.py --seeds 0,1 --e3-seeds 0 --e4-seeds 0,1 --robust-algos sac,rsac
```

### Results of the lean study

These are measured on held-out test days, as the % of the perfect-foresight LP savings captured (mean of 2 seeds):

| Method | G1 | G2 | G3 | G4 | G5 | G6 |
|---|---|---|---|---|---|---|
| SAC specialist | 78 | 37 | 92 | 97 | 67 | 81 |
| SAC generalist (one policy, all grids) | 73 | 15 | 89 | 94 | 61 | 79 |
| PPO specialist | 35 | −104 | 57 | 86 | −83 | 58 |
| Self-consumption rule | 10 | 28 | −37 | −83 | 22 | −73 |
| SAC zero-shot on unseen grid | −488 | −141 | −17 | −8 | −68 | −79 |

- **E1:** SAC captures 67–97 % of the optimum on 5 of 6 grids. G2 (flat tariff, value only from self-consumption) is the hard case, and its spread across the two seeds is large. PPO is far weaker, as in the original single-grid study.
- **E2:** a single generalist SAC with grid-descriptor inputs comes within about 5 points of the specialists everywhere except G2.
- **E3:** with only 5 training grids, policies don't extrapolate to an unseen configuration. Specialists don't transfer between grids either (`fig_e3_transfer_sac.pdf`). Generalization needs much more configuration diversity during training.
- **E4:** with 30 % forecast error, the open-loop LP collapses on real-time prices (G5: −191 %) and demand charges (G6). MPC is best on G1, G2 and G5. Domain-randomized R-SAC beats MPC on G3, G4 and G6, where demand charges and weak-grid caps punish forecast errors. Plain SAC degrades smoothly, except on G6, where it fails at high error (−163 %) unless it was trained with noise.

**Normalizer note.** Observation features that never vary in training, such as the forecast-error input without noise or a specialist's grid descriptor, are only centred, not scaled. Scaling them by their ~0 standard deviation amplified unseen variation about 10⁴×. In the first evaluation, that produced a spurious collapse of SAC at 5 % forecast error. The pre-fix evaluation is kept as `daily_results_before_normfix.csv`. E1 and E2 are unaffected to within float noise.

The 3M-step single-seed pilot is kept in `results/multigrid_pilot_3M/`.

Outputs in `results/multigrid/`:
- `daily_results.csv`: one row per evaluated day;
- `summary_*.csv`: summary tables;
- `fig_*.pdf`: figures;
- `table_*.tex`: LaTeX tables;
- `checkpoints/`: trained agents.

## Notes on performance

Training is bound by kernel-launch latency, not GPU compute: SAC fires many small
sequential operations that cannot saturate a laptop GPU. Measured on an RTX 3050,
CUDA gives roughly 2.25× over CPU (~2 h vs ~4.5 h for all three agents). Larger
batches do not help — a batch of 1024 costs the same as 256.

<img width="1109" height="480" alt="image" src="https://github.com/user-attachments/assets/1214c9e3-fb95-4f5d-836d-7ef23b24c40c" />
<img width="1057" height="352" alt="image" src="https://github.com/user-attachments/assets/22673acd-1de8-4c04-9e03-31b4196c93a4" />

