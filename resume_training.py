# resume_training.py
# Continue a pipeline run that stopped partway through.
#
# main.py always trains all three agents from scratch. This picks up where the
# previous run left off: any agent whose model already exists in results/ is
# loaded instead of retrained, then evaluation, plots and the robustness sweep
# run exactly as main.py would.
#
#   python resume_training.py
#   python resume_training.py --force rsac      # retrain rsac even if saved

import argparse
import os
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from evaluation import (
    run_comprehensive_evaluation,
    create_summary_table,
    save_evaluation_results,
)
from main import setup_environment, print_header
from microgrid_env import MicrogridEnv
from rl_agents import (
    NoopAgent,
    RuleBasedAgent,
    TrainingCallback,
    create_ppo_agent,
    create_robust_sac_agent,
    create_sac_agent,
    save_agent,
    train_agent,
)
from robustness_test import (
    RobustnessConfig,
    create_robustness_summary,
    plot_robustness_comparison,
    run_robustness_test,
    save_robustness_results,
)
from visualization import generate_all_plots

OUTPUT_DIR = "results"
SAC_TIMESTEPS = 200_000
PPO_TIMESTEPS = 200_000
N_EVAL_EPISODES = 100

DATA_PATH = "electricityConsumptionAndProductioction.csv"
FALLBACK_DATA_PATH = "PecanStreet_10_Homes_1Min_Data.csv"


def model_path(stem):
    return os.path.join(OUTPUT_DIR, f"{stem}.zip")


def have(stem):
    return os.path.exists(model_path(stem))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--force",
        nargs="*",
        default=[],
        choices=["sac", "rsac", "ppo"],
        help="retrain these even if a saved model exists",
    )
    args = ap.parse_args()
    force = set(args.force)

    print_header("RESUMING MICROGRID TRAINING PIPELINE")

    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    solar, load, prices, config = setup_environment(DATA_PATH, FALLBACK_DATA_PATH)

    plan = []
    for stem, label in (("sac_model", "SAC"), ("rsac_model", "R-SAC"), ("ppo_model", "PPO")):
        key = stem.replace("_model", "")
        plan.append((label, "retrain" if (key in force or not have(stem)) else "reuse"))

    print("\nPlan:")
    for label, what in plan:
        mark = "train" if what == "retrain" else "load saved"
        print("  %-7s %s" % (label, mark))

    trained_agents = {}
    total_train_time = 0.0

    # ---- SAC ----
    if "sac" not in force and have("sac_model"):
        print("\n[1/3] SAC - loading saved model")
        env = MicrogridEnv(solar, load, prices, config=config)
        trained_agents["SAC"] = SAC.load(model_path("sac_model"), env=env)
        print("  loaded (%s timesteps)" % f"{trained_agents['SAC'].num_timesteps:,}")
    else:
        print("\n[1/3] Training SAC...")
        env = MicrogridEnv(solar, load, prices, config=config)
        agent = create_sac_agent(env, verbose=0)
        t0 = time.time()
        train_agent(agent, SAC_TIMESTEPS, callback=TrainingCallback(log_interval=2000),
                    progress_bar=True)
        dt = time.time() - t0
        total_train_time += dt
        save_agent(agent, f"{OUTPUT_DIR}/sac_model")
        trained_agents["SAC"] = agent
        print("  SAC trained in %.1f min" % (dt / 60))

    # ---- R-SAC ----
    if "rsac" not in force and have("rsac_model"):
        print("\n[2/3] R-SAC - loading saved model")
        env = MicrogridEnv(solar, load, prices, config=config)
        trained_agents["R-SAC"] = SAC.load(model_path("rsac_model"), env=env)
        print("  loaded (%s timesteps)" % f"{trained_agents['R-SAC'].num_timesteps:,}")
    else:
        print("\n[2/3] Training R-SAC (domain randomisation)...")
        rsac_env = MicrogridEnv(
            solar, load, prices, config=config,
            randomize_env=True, noise_level=0.30,
            variable_noise_level=True, correlation=0.9,
        )
        agent, _ = create_robust_sac_agent(rsac_env, verbose=0)
        t0 = time.time()
        train_agent(agent, SAC_TIMESTEPS, callback=TrainingCallback(log_interval=2000),
                    progress_bar=True)
        dt = time.time() - t0
        total_train_time += dt
        save_agent(agent, f"{OUTPUT_DIR}/rsac_model")
        trained_agents["R-SAC"] = agent
        print("  R-SAC trained in %.1f min" % (dt / 60))

    # ---- PPO ----
    ppo_stats = f"{OUTPUT_DIR}/ppo_vec_normalize.pkl"
    if "ppo" not in force and have("ppo_model") and os.path.exists(ppo_stats):
        print("\n[3/3] PPO - loading saved model")
        venv = DummyVecEnv([lambda: MicrogridEnv(solar, load, prices, config=config)])
        trained_agents["PPO"] = PPO.load(model_path("ppo_model"), env=venv)
        print("  loaded (%s timesteps)" % f"{trained_agents['PPO'].num_timesteps:,}")
    else:
        print("\n[3/3] Training PPO...")
        venv = DummyVecEnv([lambda: MicrogridEnv(solar, load, prices, config=config)])
        venv = VecNormalize(venv, norm_obs=True, norm_reward=True, clip_obs=10.0)
        agent = create_ppo_agent(venv, verbose=0)
        t0 = time.time()
        train_agent(agent, PPO_TIMESTEPS, callback=TrainingCallback(log_interval=3000),
                    progress_bar=True)
        dt = time.time() - t0
        total_train_time += dt
        save_agent(agent, f"{OUTPUT_DIR}/ppo_model")
        venv.save(ppo_stats)
        trained_agents["PPO"] = agent
        print("  PPO trained in %.1f min" % (dt / 60))

    if total_train_time:
        print("\nTotal new training time: %.1f min" % (total_train_time / 60))

    trained_agents["NoOp"] = NoopAgent()
    trained_agents["Rule-Based"] = RuleBasedAgent()

    # ---- evaluation ----
    print_header("Evaluating All Agents")
    eval_results = run_comprehensive_evaluation(
        agents=trained_agents,
        solar_profile=solar,
        load_profile=load,
        price_profile=prices,
        n_episodes=N_EVAL_EPISODES,
        config=config,
        verbose=True,
    )
    save_evaluation_results(eval_results, f"{OUTPUT_DIR}/evaluation_results.csv")

    # ---- plots ----
    print_header("Generating Visualizations")
    generate_all_plots(eval_results, output_dir=OUTPUT_DIR, episode_idx=0)

    # ---- robustness ----
    print_header("Robustness Testing")
    rob_results = run_robustness_test(
        agents={
            "LP": None,
            "SAC": trained_agents["SAC"],
            "R-SAC": trained_agents["R-SAC"],
            "PPO": trained_agents["PPO"],
        },
        solar_profile=solar,
        load_profile=load,
        price_profile=prices,
        config=RobustnessConfig(noise_levels=[0.0, 0.05, 0.10, 0.20, 0.30], n_scenarios=10),
        env_config=config,
        verbose=True,
    )
    fig = plot_robustness_comparison(
        rob_results, save_path=f"{OUTPUT_DIR}/13_robustness_comparison.pdf"
    )
    plt.close(fig)
    save_robustness_results(rob_results, output_path=f"{OUTPUT_DIR}/robustness_summary.csv")

    # ---- summary ----
    print_header("PIPELINE COMPLETE")
    print(create_summary_table(eval_results).to_string(index=False))
    print("\nRobustness:")
    print(create_robustness_summary(rob_results).to_string(index=False))
    print("\nNext: python export_web_data.py")

    plt.close("all")


if __name__ == "__main__":
    main()
