"""
Train the DRL agent
"""

# Standard library imports
import time
import csv

# 3rd party imports
import numpy as np
import torch
import os.path
from tqdm import tqdm

# Module imports
from config import get_config
from utils import StatesGenerator, plot_training_history
from actor_critic import Actor
from rl_env import get_benchmark_rewards


if __name__ == "__main__":
    
    config, _ = get_config()
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    states_generator = StatesGenerator(config)
    agent = Actor(config)
    
    # Calculate average reward of benchmark heuristics
    nf_reward, ff_reward, ffd_reward = get_benchmark_rewards(config, states_generator)

    # Training loop
    agent_rewards = []
    pbar = tqdm(range(config.n_episodes))
    for i in pbar:
        states, states_lens, len_mask = states_generator.generate_states_batch()
        agent_loss, critic_loss, agent_reward, predicted_reward = agent.reinforce_step(
            states,
            states_lens,
            len_mask,
            verbose=False,  # True if i % 500 == 0 else False
        )
        agent_rewards.append(agent_reward)
        
        if i % 1000 == 0 and i > 0:
            agent.lr_scheduler_actor.step()
            agent.lr_scheduler_critic.step()

            plot_training_history(
                config,
                [
                    agent_rewards,
                    [nf_reward] * config.n_episodes,
                    [ff_reward] * config.n_episodes,
                    [ffd_reward] * config.n_episodes,
                ],
                ["DRL Agent + NF", "NF", "FF", "FFD"],
                outfilepath="./train_hist.png",
                moving_avg_window=100,
            )

        pbar.set_description(
            f"Agent reward: {agent_reward:.1%} | "
            f"Critic pred. reward: {predicted_reward:.1%}"
        )

    # Save key training metrics
    if not os.path.isfile("experiments.csv"):
        col_headers = list(vars(config).keys())
        col_headers.extend(["agent_reward", "nf_reward", "ff_reward", "ffd_reward"])
        with open("experiments.csv", "w") as f:
            writer = csv.writer(f)
            writer.writerow(col_headers) 
    
    with open("experiments.csv", "a") as f:
        writer = csv.writer(f)
        row_values = list(vars(config).values())
        row_values.extend([np.mean(agent_rewards[-100:]), nf_reward, ff_reward, ffd_reward])
        writer.writerow(row_values) 
    
