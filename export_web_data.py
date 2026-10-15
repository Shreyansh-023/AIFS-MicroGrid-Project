# export_web_data.py
# Exports simulation results to JSON for the web dashboard.

import json
import os
import sys
from pathlib import Path

import numpy as np

from data_loader import get_tou_prices
from microgrid_env import MicrogridEnv, MicrogridConfig, ScenarioGenerator
from rl_agents import NoopAgent, RuleBasedAgent, load_agent
from lp_solver import solve_lp_benchmark
from evaluation import run_comprehensive_evaluation, create_summary_table
from robustness_test import run_robustness_test, create_robustness_summary, RobustnessConfig

from stable_baselines3 import SAC, PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

RESULTS_DIR = "results"
OUTPUT_PATH = "web/public/data/simulation.json"
N_EVAL_EPISODES = 25
TRAJECTORY_INITIAL_SOC = 0.5


def to_list(x, nd=4):
    return [round(float(v), nd) for v in np.asarray(x).ravel()]


def build_profiles():
    """Rebuild the exact profiles main.py uses, with the same fallback order."""
    from main import setup_environment
    solar, load, prices, config = setup_environment(
        "electricityConsumptionAndProductioction.csv",
        "PecanStreet_10_Homes_1Min_Data.csv",
    )
    return np.asarray(solar), np.asarray(load), np.asarray(prices), config


def load_trained_agents(solar, load, prices, config):
    """Load whatever models exist in results/. Always include the baselines."""
    agents = {}

    sac_path = os.path.join(RESULTS_DIR, "sac_model.zip")
    if os.path.exists(sac_path):
        env = MicrogridEnv(solar, load, prices, config=config)
        agents["SAC"] = SAC.load(sac_path, env=env)

    rsac_path = os.path.join(RESULTS_DIR, "rsac_model.zip")
    if os.path.exists(rsac_path):
        env = MicrogridEnv(solar, load, prices, config=config)
        agents["R-SAC"] = SAC.load(rsac_path, env=env)

    ppo_path = os.path.join(RESULTS_DIR, "ppo_model.zip")
    if os.path.exists(ppo_path):
        env = DummyVecEnv([lambda: MicrogridEnv(solar, load, prices, config=config)])
        agents["PPO"] = PPO.load(ppo_path, env=env)

    agents["Rule-Based"] = RuleBasedAgent()
    agents["NoOp"] = NoopAgent()
    return agents


def normalized_env_for(agent_name, solar, load, prices, config):
    """PPO was trained on normalized observations, so replay it the same way."""
    import gymnasium as gym

    env = MicrogridEnv(solar, load, prices, config=config)
    if agent_name != "PPO":
        return env

    stats_path = os.path.join(RESULTS_DIR, "ppo_vec_normalize.pkl")
    if not os.path.exists(stats_path):
        return env

    dummy = DummyVecEnv([lambda: MicrogridEnv(solar, load, prices, config=config)])
    vec_norm = VecNormalize.load(stats_path, dummy)
    rms = vec_norm.obs_rms

    class Normalized(gym.Wrapper):
        def _norm(self, obs):
            return np.clip((obs - rms.mean) / np.sqrt(rms.var + 1e-8), -10, 10)

        def step(self, action):
            obs, r, term, trunc, info = self.env.step(action)
            return self._norm(obs), r, term, trunc, info

        def reset(self, **kwargs):
            obs, info = self.env.reset(**kwargs)
            return self._norm(obs), info

        def get_episode_results(self):
            return self.env.get_episode_results()

    return Normalized(env)


def trace_episode(agent, env):
    """Replay one episode and capture the per-hour trajectory."""
    obs, _ = env.reset(options={"initial_soc": TRAJECTORY_INITIAL_SOC})
    done = False
    while not done:
        action, _ = agent.predict(obs, deterministic=True)
        obs, _, term, trunc, _ = env.step(action)
        done = term or trunc

    r = env.get_episode_results()
    return {
        "soc": to_list(np.asarray(r["soc_trajectory"]) ),
        "batteryPower": to_list(r["battery_power"]),
        "gridPower": to_list(r["grid_power"]),
        "costPerStep": to_list(r["cost_per_step"]),
        "cumulativeProfit": to_list(-np.asarray(r["cumulative_cost"])),
        "totalProfit": round(float(-r["total_cost"]), 4),
        "energyProfit": round(float(-r["energy_cost"]), 4),
        "peakPenalty": round(float(r["peak_penalty"]), 4),
        "degradationCost": round(float(r["degradation_cost"]), 4),
        "throughput": round(float(r["throughput"]), 4),
        "peakViolations": int(r["peak_violations"]),
        "finalSoc": round(float(r["final_soc"]), 4),
    }


def trace_lp(solar, load, prices, config):
    lp = solve_lp_benchmark(solar, load, prices, TRAJECTORY_INITIAL_SOC)
    soc = np.asarray(lp["soc_trajectory"])
    out = {
        "soc": to_list(soc),
        "batteryPower": to_list(lp["battery_power"]),
        "gridPower": to_list(lp.get("grid_power", np.zeros(24))),
        "totalProfit": round(float(-lp["total_cost"]), 4),
        "energyProfit": round(float(-lp["energy_cost"]), 4),
        "peakPenalty": round(float(lp["peak_penalty"]), 4),
        "degradationCost": round(float(lp["degradation_cost"]), 4),
        "throughput": round(float(lp.get("throughput", 0.0)), 4),
        "finalSoc": round(float(soc[-1]), 4),
    }
    if "cumulative_cost" in lp:
        out["cumulativeProfit"] = to_list(-np.asarray(lp["cumulative_cost"]))
    return out


def main():
    solar, load, prices, config = build_profiles()
    agents = load_trained_agents(solar, load, prices, config)

    trained_names = [n for n in agents if n not in ("Rule-Based", "NoOp")]
    print("\nModels found: %s" % (trained_names or "none (baselines only)"))

    print("\nRunning evaluation (%d episodes)..." % N_EVAL_EPISODES)
    eval_results = run_comprehensive_evaluation(
        agents=agents,
        solar_profile=solar,
        load_profile=load,
        price_profile=prices,
        n_episodes=N_EVAL_EPISODES,
        config=config,
        verbose=False,
    )

    summary = create_summary_table(eval_results).to_dict(orient="records")
    summary = [
        {k: (round(float(v), 4) if isinstance(v, (int, float, np.floating)) else v)
         for k, v in row.items()}
        for row in summary
    ]

    print("Capturing per-hour trajectories...")
    trajectories = {"LP": trace_lp(solar, load, prices, config)}
    for name, agent in agents.items():
        env = normalized_env_for(name, solar, load, prices, config)
        trajectories[name] = trace_episode(agent, env)

    per_episode = {}
    for name, res in eval_results["all_results"].items():
        profits = [round(float(-e["total_cost"]), 4) for e in res["episodes"]]
        entry = {"profits": profits}
        if "gaps" in res:
            entry["gaps"] = [round(float(g), 3) for g in res["gaps"]]
        per_episode[name] = entry

    robustness = None
    if trained_names:
        print("Running robustness sweep...")
        rob_agents = {"LP": None}
        for n in trained_names:
            rob_agents[n] = agents[n]

        rob_results = run_robustness_test(
            agents=rob_agents,
            solar_profile=solar,
            load_profile=load,
            price_profile=prices,
            config=RobustnessConfig(noise_levels=[0.0, 0.05, 0.10, 0.20, 0.30], n_scenarios=10),
            env_config=config,
            verbose=False,
        )

        noise_levels = rob_results["noise_levels"]
        curves = {}
        for name, agt in rob_results["agents"].items():
            pts = []
            for i, noise in enumerate(noise_levels):
                profits = np.asarray(agt["all_profits"][i], dtype=float)
                pts.append({
                    "noise": round(float(noise) * 100, 1),
                    "meanProfit": round(float(agt["mean_profits"][i]), 4),
                    "stdProfit": round(float(agt["std_profits"][i]), 4),
                    "minProfit": round(float(profits.min()), 4),
                    "maxProfit": round(float(profits.max()), 4),
                })
            curves[name] = pts

        rob_summary = create_robustness_summary(rob_results).to_dict(orient="records")
        rob_summary = [
            {k: (round(float(v), 3) if isinstance(v, (int, float, np.floating)) else v)
             for k, v in row.items()}
            for row in rob_summary
        ]
        robustness = {"curves": curves, "summary": rob_summary}

    # forecast-vs-actual illustration at 30% noise
    gen = ScenarioGenerator(seed=7)
    noisy_solar = gen.add_noise(solar, 0.30, 0.9, clip_min=0.0)
    noisy_load = gen.add_noise(load, 0.30, 0.9, clip_min=0.1)

    payload = {
        "meta": {
            "evalEpisodes": N_EVAL_EPISODES,
            "trajectoryInitialSoc": TRAJECTORY_INITIAL_SOC,
            "agentsTrained": trained_names,
            "config": {
                "batteryCapacityKwh": config.e_max,
                "maxPowerKw": config.p_bat_max,
                "efficiency": config.eta_charge,
                "rampRateKwPerH": config.ramp_rate,
                "peakThresholdKw": config.p_grid_peak,
                "peakPenaltyRate": config.peak_penalty_rate,
                "degradationCostPerKwh": config.degradation_cost_per_kwh,
                "forecastHorizonH": config.forecast_horizon,
                "minSocRatio": config.e_min_ratio,
            },
        },
        "profiles": {
            "hours": list(range(24)),
            "solar": to_list(solar),
            "load": to_list(load),
            "netLoad": to_list(np.asarray(load) - np.asarray(solar)),
            "price": to_list(prices),
            "forecastDemo": {
                "noiseLevel": 30,
                "actualSolar": to_list(noisy_solar),
                "actualLoad": to_list(noisy_load),
            },
        },
        "summary": summary,
        "trajectories": trajectories,
        "perEpisode": per_episode,
        "robustness": robustness,
    }

    out = Path(OUTPUT_PATH)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("\nWrote %s" % out)
    print("Agents exported: %s" % ", ".join(trajectories.keys()))


if __name__ == "__main__":
    main()
