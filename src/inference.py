import torch
from actor_critic import Actor
from rl_env import StatesGenerator, get_benchmark_rewards
import json

@torch.inference_mode()
def inference(config):
    # Load model
    device = config.device if torch.cuda.is_available() else "cpu"
    actor = Actor(config)
    actor.policy_dnn = torch.load(config.model_path, map_location=torch.device(device))    

    # Generate states
    if config.inference_data_path:
        with open(config.inference_data_path, "r") as f:
            states_batch = json.load(f) 
    
    states_generator = StatesGenerator(config)
    states_batch, states_lens, len_mask = states_generator.generate_states_batch()
    
    # Get agent reward
    agent_reward, _ = actor.apply_policy(
        states_batch,
        states_lens,
        len_mask
    )
    print(f"Agent reward: {agent_reward:.1%}")