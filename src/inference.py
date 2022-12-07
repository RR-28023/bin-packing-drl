import torch
from actor_critic import Actor
from rl_env import StatesGenerator, get_benchmark_rewards, compute_reward
import ast
import numpy as np

import time
import gc

@torch.inference_mode()
def inference(config):

    # Generate states
    if config.inference_data_path:
        print("Opening file {config.inference_data_path}")
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
        print("Generating set of tasks")
        states_generator = StatesGenerator(config)
        states_batch, states_lens, len_mask = states_generator.generate_states_batch()
    
    # Load model
    config.batch_size = len(states_batch)
    device = config.device if torch.cuda.is_available() else "cpu"
    actor = Actor(config)
    dec_input = actor.policy_dnn.dec_input
    actor.policy_dnn = torch.load(config.model_path, map_location=torch.device(device))
    actor.policy_dnn.dec_input = dec_input

    # Execute model  
    for _ in range(3): # Warmup
        allocation_order, elapsed_time = actor.apply_policy(
            states_batch,
            states_lens,
            len_mask
        )
    # gc.disable()
    start = time.process_time_ns()
    allocation_order, elapsed_time = actor.apply_policy(
        states_batch,
        states_lens,
        len_mask
    )
    middle = time.process_time_ns()
    middle_v2 = time.time()
    avg_occ_ratio = compute_reward(config, states_batch, len_mask, allocation_order).mean()
    end = time.process_time_ns()
    end_v2 = time.time()
    # gc.enable()

    model_mean_time_ms = (end - start) / len(states_batch) / 1000000
    drl_mean_time_ms   = (middle - start) / len(states_batch) / 1000000
    drl_mean_time_ms_v2   = elapsed_time * 1e3 / len(states_batch)
    order_mean_time_ms = (end - middle) / len(states_batch) / 1000000
    order_mean_time_ms_v2 = (end_v2 - middle_v2) * 1e3 / len(states_batch)

    str = f"Average occupancy ratio with RL+{config.agent_heuristic} agent: {avg_occ_ratio:.1%} "
    str += f"({model_mean_time_ms} ms = {drl_mean_time_ms} ms + {order_mean_time_ms} ms)"
    print(str)
    print(f"\nVersion with time.time():")
    str = f"Average occupancy ratio with RL+{config.agent_heuristic} agent: {avg_occ_ratio:.1%} "
    str += f"({drl_mean_time_ms_v2 + order_mean_time_ms_v2} ms = {drl_mean_time_ms_v2} ms + {order_mean_time_ms_v2} ms)\n"
    print(str)

    # Execute heuristics
    benchmark_rewards = get_benchmark_rewards(config, states_batch=states_batch)
