import torch
from actor_critic import Actor
from rl_env import StatesGenerator, get_benchmark_rewards, compute_reward
import ast
import numpy as np

@torch.inference_mode()
def inference(config):

    # Generate states
    if config.inference_data_path:
        with open(config.inference_data_path, "r") as f:
            states_batch = [ast.literal_eval(line.rstrip()) for line in f]
        
        # Generate len mask and lens list
        states_lens = np.array([len(state) for state in states_batch])
        len_mask = np.array([[1]*l + [0]*(config.max_num_items - l) for l in states_lens])
        # Pad to max length
        states_batch = np.array([
            [*state, *[0]*(config.max_num_items - len(state))] 
            for state in states_batch
        ])

    else:
        states_generator = StatesGenerator(config)
        states_batch, states_lens, len_mask = states_generator.generate_states_batch()
    
    # Load model
    config.batch_size = len(states_batch)
    device = config.device if torch.cuda.is_available() else "cpu"
    actor = Actor(config)
    actor.policy_dnn = torch.load(config.model_path, map_location=torch.device(device))
    actor.policy_dnn.dec_input = actor.policy_dnn.dec_input[:config.batch_size]    

    # Get agent reward
    allocation_order = actor.apply_policy(
        states_batch,
        states_lens,
        len_mask
    )
    avg_occ_ratio = compute_reward(config, states_batch, len_mask, allocation_order).mean()
    benchmark_rewards = get_benchmark_rewards(config, states_batch=states_batch)
    print(f"Average occupancy ratio with RL agent: {avg_occ_ratio:.1%}")
    for reward, heuristic in zip(benchmark_rewards, ("NF", "FF", "FFD")):
        print(f"Average occupancy ratio with {heuristic} heuristic: {reward:.1%}")
