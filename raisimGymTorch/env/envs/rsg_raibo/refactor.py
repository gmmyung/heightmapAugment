import os
import math
import time
import datetime
import argparse

import torch
import torch.nn as nn
import numpy as np

from ruamel.yaml import YAML
from ruamel.yaml.compat import StringIO

from raisimGymTorch.env.RaisimGymVecEnv import RaisimGymVecEnv as VecEnv
from raisimGymTorch.env.bin.rsg_raibo import RaisimGymEnv, NormalSampler
from raisimGymTorch.env.RewardAnalyzer import RewardAnalyzer
from foothold import FootHoldPredictor

from raisimGymTorch.helper.raisim_gym_helper import (
    ConfigurationSaver,
    load_param,
    tensorboard_launcher,
)
import raisimGymTorch.algo.ppo.ppo as PPO
import raisimGymTorch.algo.ppo.module as ppo_module


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m", "--mode", help="Set mode either train or test or retrain", 
        type=str, default="train"
    )
    parser.add_argument(
        "-w", "--weight", help="Path to pre-trained weight", 
        type=str, default=""
    )
    return parser.parse_args()


def create_environment(cfg, home_path):
    """Create the VecEnv environment based on provided configuration."""
    string_io = StringIO()
    YAML().dump(cfg["environment"], string_io)
    env = VecEnv(RaisimGymEnv(os.path.join(home_path, "rsc"), string_io.getvalue()))
    env.seed(cfg["seed"])
    return env


def create_actor_critic(cfg, ob_dim, act_dim, device):
    """Create Actor and Critic modules for PPO."""
    # Actor
    actor_arch = ppo_module.MLP(
        cfg["architecture"]["policy_net"], nn.LeakyReLU, ob_dim, act_dim
    )
    actor_dist = ppo_module.MultivariateGaussianDiagonalCovariance(
        act_dim,
        cfg["environment"]["num_envs"],
        init_std=1.0,
        fast_sampler=NormalSampler(act_dim),
        seed=cfg["seed"],
    )
    actor = ppo_module.Actor(actor_arch, actor_dist, device)

    # Critic
    critic_arch = ppo_module.MLP(
        cfg["architecture"]["value_net"], nn.LeakyReLU, ob_dim, 1
    )
    critic = ppo_module.Critic(critic_arch, device)

    return actor, critic


def save_model(update, saver, actor, critic, optimizer):
    """Save model checkpoints (actor, critic, optimizer) at a given iteration."""
    save_path = os.path.join(saver.data_dir, f"full_{update}.pt")
    torch.save(
        {
            "actor_architecture_state_dict": actor.architecture.state_dict(),
            "actor_distribution_state_dict": actor.distribution.state_dict(),
            "critic_architecture_state_dict": critic.architecture.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        save_path,
    )
    return save_path


def load_graph_for_evaluation(cfg, ob_dim, act_dim, model_path):
    """Load a new MLP for evaluation from a saved checkpoint."""
    loaded_graph = ppo_module.MLP(cfg["architecture"]["policy_net"], nn.LeakyReLU, ob_dim, act_dim)
    checkpoint = torch.load(model_path)
    loaded_graph.load_state_dict(checkpoint["actor_architecture_state_dict"])
    return loaded_graph


def evaluate_policy(env, ppo, loaded_graph, reward_analyzer, cfg, update):
    """Evaluate the current policy by visualizing, recording a video, and collecting reward info."""
    env.turn_on_visualization()
    video_filename = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + f"policy_{update}.mp4"
    env.start_video_recording(video_filename)

    n_steps = math.floor(cfg["environment"]["max_time"] / cfg["environment"]["control_dt"])
    for _ in range(n_steps):
        with torch.no_grad():
            frame_start = time.time()
            obs = env.observe(False)
            action = loaded_graph.architecture(torch.from_numpy(obs).cpu())
            reward, dones = env.step(action.cpu().detach().numpy())
            reward_analyzer.add_reward_info(env.get_reward_info())
            frame_time = time.time() - frame_start
            wait_time = cfg["environment"]["control_dt"] - frame_time
            if wait_time > 0.0:
                time.sleep(wait_time)

    env.stop_video_recording()
    env.turn_off_visualization()

    reward_analyzer.analyze_and_plot(update)
    env.reset()


def main():
    # ----------------
    # Initialization
    # ----------------
    args = parse_arguments()
    mode = args.mode
    weight_path = args.weight

    # Check device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Directories and paths
    task_name = "raibo2_locomotion"
    task_path = os.path.dirname(os.path.realpath(__file__))
    home_path = os.path.join(task_path, "../../../..")

    # Load config
    cfg = YAML().load(open(os.path.join(task_path, "cfg.yaml"), "r"))

    # Create environment
    env = create_environment(cfg, home_path)
    ob_dim = env.num_obs
    act_dim = env.num_acts

    # Steps and other parameters
    n_steps = math.floor(cfg["environment"]["max_time"] / cfg["environment"]["control_dt"])
    total_steps = n_steps * env.num_envs

    # Create actor and critic
    actor, critic = create_actor_critic(cfg, ob_dim, act_dim, device)

    # Configuration Saver
    saver = ConfigurationSaver(
        log_dir=os.path.join(home_path, "raisimGymTorch", "data", task_name),
        save_items=[
            os.path.join(task_path, "cfg.yaml"),
            os.path.join(task_path, "Environment.hpp"),
        ],
    )
    tensorboard_launcher(os.path.join(saver.data_dir, ".."))

    # Create PPO object
    ppo = PPO.PPO(
        actor=actor,
        critic=critic,
        num_envs=cfg["environment"]["num_envs"],
        num_transitions_per_env=n_steps,
        num_learning_epochs=4,
        gamma=0.996,
        lam=0.95,
        num_mini_batches=4,
        device=device,
        log_dir=saver.data_dir,
        shuffle_batch=False,
    )

    # Reward Analyzer
    reward_analyzer = RewardAnalyzer(env, ppo.writer)

    # Foothold Predictor
    foothold_predictor = FootHoldPredictor(cfg)

    # Optionally load weights if retraining
    if mode == "retrain" and weight_path:
        load_param(weight_path, env, actor, critic, ppo.optimizer, saver.data_dir)

    # ----------------
    # Training loop
    # ----------------
    avg_rewards = []
    for update in range(1_000_000):
        start_time = time.time()
        env.reset()
        total_reward = 0
        total_done = 0

        # Evaluate current policy periodically
        if update % cfg["environment"]["eval_every_n"] == 0:
            print("Visualizing and evaluating the current policy")
            model_path = save_model(update, saver, actor, critic, ppo.optimizer)

            # Load a separate graph for demonstration of save/load
            loaded_graph = load_graph_for_evaluation(cfg, ob_dim, act_dim, model_path)
            evaluate_policy(env, ppo, loaded_graph, reward_analyzer, cfg, update)
            env.save_scaling(saver.data_dir, str(update))


        # Collect experience
        for step in range(n_steps):
            obs = env.observe()
            action = ppo.act(obs)
            reward, dones = env.step(action)
            foothold_predictor.step(env.get_footholds())
            ppo.step(value_obs=obs, rews=reward, dones=dones)
            total_done += np.sum(dones)
            total_reward += np.sum(reward)
        

        if update % 10 == 0:
            foothold_predictor.flatten_footholds()
            foothold_predictor.train_lstm(epochs=20)
            foothold_predictor.reset()


        # Final step for advantage/value calculation
        last_obs = env.observe()
        ppo.update(
            actor_obs=last_obs,
            value_obs=last_obs,
            log_this_iteration=(update % 10 == 0),
            update=update,
        )

        # Compute average performance
        avg_performance = total_reward / total_steps
        avg_dones = total_done / total_steps
        avg_rewards.append(avg_performance)

        # Update actor distribution parameters if needed
        actor.update()
        actor.distribution.enforce_minimum_std((torch.ones(12) * 0.2).to(device))

        # Curriculum callback in environment
        env.curriculum_callback()

        # ----------------
        # Logging
        # ----------------
        iteration_time = time.time() - start_time
        fps = total_steps / iteration_time
        real_time_factor = fps * cfg["environment"]["control_dt"]

        print("----------------------------------------------------")
        print(f"{update:>6}th iteration")
        print(f"{'average ll reward:':<40} {avg_performance:0.10f}")
        print(f"{'dones:':<40} {avg_dones:0.6f}")
        print(f"{'time elapsed in this iteration:':<40} {iteration_time:6.4f}")
        print(f"{'fps:':<40} {fps:6.0f}")
        print(f"{'real time factor:':<40} {real_time_factor:6.0f}")
        print("----------------------------------------------------\n")


if __name__ == "__main__":
    main()

