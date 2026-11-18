# export_multigrid_web.py
# Multi-grid results -> JSON for the web dashboard (one file per grid + an index).
#
#   python export_multigrid_web.py
#
# Writes web/public/data/grids/<GID>.json in the same schema as
# web/public/data/simulation.json, so every dashboard section can switch grid,
# plus web/public/data/grids/index.json with the cross-grid tables.
#
# Inputs: results/multigrid/daily_results.csv (lean study) and the trained
# checkpoints in results/multigrid/checkpoints/. Nothing is retrained.

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from multigrid import experiments as X
from multigrid.grids import GRIDS, GRID_IDS, finite
from multigrid.lp import grid_params, solve_schedule
from multigrid.torch_env import BatchedMicrogridEnv, T, OBS_DIM

OUT = Path("web/public/data/grids")
SEED = 0                 # checkpoint used for the replayed day
INIT_SOC = 0.5
NOISE_LEVELS = [0.0, 0.05, 0.10, 0.20, 0.30]
BIG = 1e8

# method label in daily_results.csv -> dashboard label
E1_METHODS = {"LP (oracle)": "LP", "MPC": "MPC", "SAC": "SAC", "PPO": "PPO",
              "Self-Cons.": "Self-Cons.", "Rule-Based": "Rule-Based", "NoOp": "NoOp"}
ROB_METHODS = {"LP (open loop)": "LP", "MPC": "MPC", "SAC": "SAC", "R-SAC": "R-SAC"}

LOAD_LABEL = {"G1": "Household load", "G2": "Household load", "G3": "Office load",
              "G4": "Community load", "G5": "Community + EV load", "G6": "Warehouse load"}
TARIFF_LABEL = {
    "tou_residential": "Residential TOU",
    "flat": "Flat rate",
    "tou_commercial": "Commercial TOU + demand charge",
    "tou_industrial": "Industrial TOU + demand charge",
    "rtp_nyiso": "Real-time (NYISO day-ahead LBMP)",
}
STRATEGY = {
    "G1": "Buy off-peak, store midday solar, and sell into the 14:00–21:00 on-peak window "
          "while keeping grid draw under the 10 kW soft cap.",
    "G2": "The price is flat, so arbitrage is worthless. The only value is storing solar that "
          "would otherwise be exported at 25% of retail and using it in the evening.",
    "G3": "Shave the office's daily peak, since every kW of peak import costs $0.60 per day, "
          "and shift solar into the 16:00–21:00 on-peak price.",
    "G4": "The feeder can import only 30 kW. The battery must pre-charge so it can cover "
          "evening load above the cap, because unserved energy costs $5/kWh.",
    "G5": "Prices follow the NYISO wholesale market hour by hour and nothing can be exported. "
          "The battery charges in cheap hours and covers evening EV charging.",
    "G6": "With a $1.00/kW-day demand charge and a slow 40 kW/h ramp, the battery must start "
          "shaving the warehouse peak early.",
}
BLURBS = {
    "LP": "Linear program with perfect foresight of the actual day. Its objective is exactly the "
          "environment's daily cost, so it is the true optimum.",
    "MPC": "Model-predictive control: re-solves the LP every hour on the forecast and applies the "
           "first action. This is the closed-loop baseline a utility would deploy.",
    "SAC": "Soft Actor-Critic specialist trained on this grid only: GPU-batched env, 6M steps, "
           "with the best checkpoint picked on validation days.",
    "Generalist SAC": "One SAC policy trained on all six grids at once. It reads a 9-feature "
                      "grid descriptor to know which grid it is controlling.",
    "R-SAC": "SAC trained under domain randomisation: each episode draws forecast error up to 30%.",
    "PPO": "Proximal Policy Optimisation specialist with a tanh-squashed Gaussian, trained on "
           "512 parallel envs for 10M steps.",
    "Zero-shot SAC": "SAC trained on the other five grids and never shown this one "
                     "(leave-one-grid-out transfer test).",
    "Self-Cons.": "Self-consumption heuristic: charge on surplus PV, discharge to cover load, "
                  "and steer back to 50% SOC over the last four hours.",
    "Rule-Based": "The original price-threshold heuristic from the single-home study.",
    "NoOp": "Battery idle for all 24 hours. This is the reference for savings.",
}


def r4(x):
    return round(float(x), 4)


def to_list(x, nd=4):
    return [round(float(v), nd) for v in np.asarray(x).ravel()]


# ------------------------------------------------------------- tables --

def load_daily():
    d = pd.read_csv(X.DAILY, low_memory=False)
    d["train_tag"] = d["train_tag"].fillna("").astype(str)
    return d


def per_day(d, exp, gid, method, noise=0.0, scen=None):
    """Seed-averaged cost per test day (Series indexed by day)."""
    m = (d.experiment == exp) & (d.grid == gid) & (d.method == method) & np.isclose(d.noise, noise)
    if scen is not None:
        m &= d.scenario == scen
    sub = d[m]
    return sub.groupby("day")[[c for c in sub.columns if c not in
                               ("experiment", "grid", "method", "train_tag", "day")]].mean()


def method_tables(d, gid):
    """{label: DataFrame per day} for the noise-free comparison on test days."""
    out = {}
    for raw, lab in E1_METHODS.items():
        out[lab] = per_day(d, "E1", gid, raw)
    out["Generalist SAC"] = per_day(d, "E2", gid, "Generalist SAC")
    out["R-SAC"] = per_day(d, "E4", gid, "R-SAC", 0.0, scen=0)
    out["Zero-shot SAC"] = per_day(d, "E3", gid, "Zero-shot SAC")
    return {k: v for k, v in out.items() if len(v)}


def capture(m_cost, noop_cost, lp_cost):
    den = noop_cost - lp_cost
    return float((noop_cost - m_cost) / den * 100) if abs(den) > 1e-9 else float("nan")


def summary_rows(tabs):
    lp, noop = tabs["LP"]["cost"].mean(), tabs["NoOp"]["cost"].mean()
    rows = []
    for lab, df in tabs.items():
        c = df["cost"]
        rows.append({
            "Agent": lab,
            "Mean Profit ($)": r4(-c.mean()),
            "Std Profit ($)": r4(c.std()),
            "Gap vs LP (%)": r4((c.mean() - lp) / abs(lp) * 100) if abs(lp) > 1e-9 else 0.0,
            "Capture (%)": round(capture(c.mean(), noop, lp), 1),
            "Mean Throughput (kWh)": r4(df["throughput"].mean()),
            "Mean Final SOC": r4(df["final_soc"].mean()),
            "Mean Unserved (kWh)": r4(df["unserved_kwh"].mean()),
            "Mean Demand Charge ($)": r4(df["demand"].mean()),
        })
    return rows


def robustness(d, gid):
    curves, cap = {}, {}
    base = d[(d.experiment == "E4") & (d.grid == gid)]
    runs = lambda meth: (base[base.method == meth]
                         .groupby(["noise", "seed", "scenario"])["cost"].mean())
    # capture is measured against the 0%-error oracle and NoOp of the same days,
    # as in multigrid/report.py (robustness_table), so it matches the paper.
    at0 = lambda r: float(r[np.isclose(r.index.get_level_values(0), 0.0)].mean())
    o, z = at0(runs("LP (oracle)")), at0(runs("NoOp"))
    for raw, lab in ROB_METHODS.items():
        r = runs(raw)
        if not len(r):
            continue
        pts, cps = [], []
        for n in NOISE_LEVELS:
            v = -r[np.isclose(r.index.get_level_values(0), n)].values
            if not len(v):
                continue
            pts.append({"noise": round(n * 100, 1), "meanProfit": r4(v.mean()),
                        "stdProfit": r4(v.std()), "minProfit": r4(v.min()), "maxProfit": r4(v.max())})
            cps.append({"noise": round(n * 100, 1), "capture": round(capture(-v.mean(), z, o), 1),
                        "std": round(float(v.std() / abs(z - o) * 100), 1)})
        curves[lab], cap[lab] = pts, cps
    summ = []
    for lab, pts in curves.items():
        p0, p30 = pts[0]["meanProfit"], pts[-1]["meanProfit"]
        deg = (p0 - p30) / abs(p0) * 100 if abs(p0) > 1e-9 else 0.0
        summ.append({"Agent": lab, "Profit @ 0% Noise": round(p0, 3), "Profit @ 30% Noise": round(p30, 3),
                     "Capture @ 30% (%)": cap[lab][-1]["capture"],
                     "Degradation (%)": round(deg, 3), "Robustness Score": round(100 - abs(deg), 3)})
    summ.sort(key=lambda r: -r["Capture @ 30% (%)"])
    n_runs = int(base[(base.method == "SAC")].groupby(["seed", "scenario"]).ngroups)
    return {"curves": curves, "capture": cap, "summary": summ}, n_runs


# ------------------------------------------------------- trajectories --

def pick_day(tabs):
    """A representative test day: above-median LP savings, SAC capture closest to its mean."""
    sav = tabs["NoOp"]["cost"] - tabs["LP"]["cost"]
    sac = tabs["SAC"]["cost"]
    target = capture(sac.mean(), tabs["NoOp"]["cost"].mean(), tabs["LP"]["cost"].mean())
    lo, hi = sav.quantile(0.5), sav.quantile(0.8)
    cand = sav[(sav >= lo) & (sav <= hi)].index
    day_cap = (tabs["NoOp"]["cost"][cand] - sac[cand]) / sav[cand] * 100
    return int((day_cap - target).abs().idxmin())


def single_env(bank, day, noise=0.0, scen=0):
    env = BatchedMicrogridEnv(bank, 1, split="test", scale_reward=False)
    env.gen.manual_seed(10_000 + scen)
    obs = env.reset(day_idx=[day], init_soc=torch.full((1,), INIT_SOC, device=env.device),
                    noise_level=noise)
    return env, obs


def trace(bank, gid, day, controller):
    env, obs = single_env(bank, day)
    prm = grid_params(bank.configs[gid])
    pbat = float(env.p["p_bat_max"][0])
    arr = lambda x, h=0: x[0, h:].cpu().numpy().astype(float)
    with torch.no_grad():
        if controller == "LP":
            plan = solve_schedule(prm, arr(env.a_load), arr(env.a_pv), arr(env.a_pimp), arr(env.a_pexp),
                                  float(env.soc[0]))["battery_power"]
            for t in range(T):
                env.step(torch.tensor([plan[t] / pbat]))
        elif controller == "MPC":
            for t in range(T):
                plan = solve_schedule(prm, arr(env.f_load, t), arr(env.f_pv, t), arr(env.f_pimp, t),
                                      arr(env.f_pexp, t), float(env.soc[0]), float(env.prev_bp[0]),
                                      float(env.run_max_imp[0]))["battery_power"]
                env.step(torch.tensor([plan[0] / pbat]))
        else:
            for t in range(T):
                obs, _, _, _ = env.step(controller.act(obs))
    tr = {k: v[0].cpu().numpy().astype(float) for k, v in env.traj.items()}
    tot = {k: float(v[0]) for k, v in env.tot.items()}
    cost = tr["cost"].copy()
    cost[-1] += tot["terminal"]
    return {
        "soc": to_list(tr["soc"]),
        "batteryPower": to_list(tr["bat"]),
        "gridPower": to_list(tr["grid"]),
        "costPerStep": to_list(cost),
        "cumulativeProfit": to_list(-np.cumsum(cost)),
        "totalProfit": r4(-tot["cost"]),
        "energyProfit": r4(-tot["energy"]),
        "peakPenalty": r4(tot["peak"] + tot["demand"]),
        "degradationCost": r4(tot["degradation"]),
        "unservedCost": r4(tot["unserved_cost"]),
        "terminalCost": r4(tot["terminal"]),
        "unservedKwh": r4(tot["unserved_kwh"]),
        "curtailedKwh": r4(tot["curtailed_kwh"]),
        "throughput": r4(tot["throughput"]),
        "peakViolations": int(tot["peak_violations"]),
        "finalSoc": r4(float(env.soc[0])),
    }


def controllers(gid):
    ag = {"LP": "LP", "MPC": "MPC"}
    for algo, tag, lab in (("sac", gid, "SAC"), ("sac", "ALL", "Generalist SAC"), ("rsac", gid, "R-SAC"),
                           ("ppo", gid, "PPO"), ("sac", "LOO-" + gid, "Zero-shot SAC")):
        if X.ckpt_path(algo, tag, SEED).exists():
            ag[lab] = X.load_agent(algo, tag, SEED)
    ag.update(X.HEURISTICS)
    return ag


# ------------------------------------------------------------- bands --

TIER = [("Off-peak", "#3fbf87"), ("Mid-peak", "#e8a33c"), ("On-peak", "#e8576b")]


def price_bands(price, rtp):
    """Contiguous price segments labelled by tier (drives chart shading)."""
    price = np.asarray(price, float)
    if np.ptp(price) < 1e-6:
        return [{"from": 0, "to": 24, "label": "Flat rate", "price": r4(price[0]), "color": "#9eb0c2"}]
    if rtp:
        q1, q2 = np.quantile(price, [1 / 3, 2 / 3])
        tier = np.where(price <= q1, 0, np.where(price <= q2, 1, 2))
        names = ["Low price", "Mid price", "High price"]
    else:
        levels = sorted(set(np.round(price, 4)))
        rank = {v: i for i, v in enumerate(levels)}
        k = len(levels)
        tier = np.array([0 if k == 1 else round(rank[round(p, 4)] * 2 / (k - 1)) for p in price])
        names = [t[0] for t in TIER]
    bands, start = [], 0
    for h in range(1, 25):
        if h == 24 or tier[h] != tier[start]:
            t = int(tier[start])
            bands.append({"from": start, "to": h, "label": names[t], "price": r4(price[start:h].mean()),
                          "color": TIER[t][1], "peak": t == 2})
            start = h
    return bands


# -------------------------------------------------------------- main --

def grid_meta(cfg, bank, gid, day, n_days, n_runs):
    date = str(bank.dates[day])
    return {
        "id": gid, "name": cfg.name, "description": cfg.description,
        "strategy": STRATEGY[gid], "loadLabel": LOAD_LABEL[gid],
        "tariff": TARIFF_LABEL[cfg.tariff], "pvKwp": cfg.pv_kwp,
        "exportRatio": cfg.export_ratio,
        "demandChargePerKwDay": cfg.demand_charge_per_kw_day,
        "importCapKw": None if finite(cfg.import_cap_kw, BIG) >= BIG else cfg.import_cap_kw,
        "exportCapKw": None if finite(cfg.export_cap_kw, BIG) >= BIG else cfg.export_cap_kw,
        "voll": cfg.voll,
        "terminalPenaltyPerUnitSoc": r4(cfg.terminal_penalty_per_kwh * cfg.e_max),
        "replayDate": date[:10], "nTestDays": n_days, "robustnessRuns": n_runs,
        "obsDim": OBS_DIM, "seeds": 2,
    }


def export_grid(d, bank, gid):
    cfg = GRIDS[gid]
    tabs = method_tables(d, gid)
    day = pick_day(tabs)
    print("  %s: replay day %d (%s)" % (gid, day, str(bank.dates[day])[:10]))

    trajectories = {lab: trace(bank, gid, day, c) for lab, c in controllers(gid).items()}
    rob, n_runs = robustness(d, gid)

    env, _ = single_env(bank, day)
    load, pv = env.f_load[0].cpu().numpy(), env.f_pv[0].cpu().numpy()
    pimp, pexp = env.f_pimp[0].cpu().numpy(), env.f_pexp[0].cpu().numpy()
    noisy, _ = single_env(bank, day, noise=0.30, scen=7)

    payload = {
        "meta": {
            "evalEpisodes": int(len(tabs["LP"])),
            "trajectoryInitialSoc": INIT_SOC,
            "agentsTrained": [a for a in ("SAC", "Generalist SAC", "R-SAC", "PPO", "Zero-shot SAC")
                              if a in trajectories],
            "config": {
                "batteryCapacityKwh": cfg.e_max, "maxPowerKw": cfg.p_bat_max,
                "efficiency": cfg.eta_charge, "rampRateKwPerH": cfg.ramp_rate,
                "peakThresholdKw": None if finite(cfg.p_grid_peak, BIG) >= BIG else cfg.p_grid_peak,
                "peakPenaltyRate": cfg.peak_penalty_rate,
                "degradationCostPerKwh": cfg.degradation_cost_per_kwh,
                "forecastHorizonH": 24, "minSocRatio": cfg.e_min_ratio,
            },
            "grid": grid_meta(cfg, bank, gid, day, int(len(tabs["LP"])), n_runs),
            "blurbs": BLURBS,
        },
        "profiles": {
            "hours": list(range(24)),
            "solar": to_list(pv), "load": to_list(load), "netLoad": to_list(load - pv),
            "price": to_list(pimp), "priceExport": to_list(pexp),
            "bands": price_bands(pimp, cfg.tariff.startswith("rtp")),
            "forecastDemo": {"noiseLevel": 30,
                             "actualSolar": to_list(noisy.a_pv[0].cpu().numpy()),
                             "actualLoad": to_list(noisy.a_load[0].cpu().numpy())},
        },
        "summary": summary_rows(tabs),
        "trajectories": trajectories,
        "perEpisode": {lab: {"profits": to_list(-df["cost"].values, 3)} for lab, df in tabs.items()},
        "robustness": rob,
    }
    (OUT / ("%s.json" % gid)).write_text(json.dumps(payload, indent=1), encoding="utf-8")

    caps = {r["Agent"]: r["Capture (%)"] for r in payload["summary"]}
    cap30 = {r["Agent"]: r["Capture @ 30% (%)"] for r in rob["summary"]}
    return {
        "id": gid, "name": cfg.name, "description": cfg.description,
        "batteryKwh": cfg.e_max, "batteryKw": cfg.p_bat_max, "pvKwp": cfg.pv_kwp,
        "tariff": TARIFF_LABEL[cfg.tariff], "loadLabel": LOAD_LABEL[gid],
        "capture": caps, "capture30": cap30,
    }


def transfer_matrix(d):
    t = d[d.experiment == "E3T"]
    lp = d[(d.experiment == "E1") & (d.method == "LP (oracle)")].groupby("grid")["cost"].mean()
    noop = d[(d.experiment == "E1") & (d.method == "NoOp")].groupby("grid")["cost"].mean()
    m = t.groupby(["train_tag", "grid"])["cost"].mean()
    return {tr: {te: round(capture(m[(tr, te)], noop[te], lp[te]), 1) for te in GRID_IDS if (tr, te) in m}
            for tr in GRID_IDS}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    d = load_daily()
    bank = X.global_bank()
    print("Exporting grids to %s" % OUT)
    grids = [export_grid(d, bank, g) for g in GRID_IDS]
    index = {
        "grids": grids,
        "methods": ["SAC", "Generalist SAC", "R-SAC", "PPO", "MPC", "Self-Cons.", "Rule-Based",
                    "Zero-shot SAC"],
        "robustMethods": ["LP", "MPC", "SAC", "R-SAC"],
        "transfer": transfer_matrix(d),
        "note": "Share of the perfect-foresight LP's savings over an idle battery captured on held-out "
                "test days (100 = optimal, 0 = no better than idle). Mean of 2 training seeds.",
    }
    (OUT / "index.json").write_text(json.dumps(index, indent=1), encoding="utf-8")
    print("Wrote %d grid files + index.json" % len(grids))
    for g in grids:
        print("  %s  SAC %5.1f  Gen %5.1f  PPO %6.1f  MPC %5.1f" % (
            g["id"], g["capture"].get("SAC", np.nan), g["capture"].get("Generalist SAC", np.nan),
            g["capture"].get("PPO", np.nan), g["capture"].get("MPC", np.nan)))


if __name__ == "__main__":
    main()
