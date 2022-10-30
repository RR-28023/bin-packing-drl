import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
from rl_env import  compute_reward
from dnns import ActorPointerNetwork, CriticNetwork
from argparse import Namespace


class Actor:
    def __init__(self, config: Namespace):

        self.batch_size = config.batch_size
        self.max_len = config.max_num_items
        self.device = config.device if torch.cuda.is_available() else "cpu"
        self.config = config
        
        # Initialize actor network and training artifacts
        self.policy_dnn = ActorPointerNetwork(config)
        if not config.inference:
            self.optimizer_actor = optim.Adam(self.policy_dnn.parameters(), lr=config.lr)
            self.lr_scheduler_actor = optim.lr_scheduler.ExponentialLR(
                self.optimizer_actor, 0.9, verbose=True
            )
        self.policy_dnn.to(self.device)

        # Initialize critic network and training artifacts
        self.critic_dnn = CriticNetwork(config)
        if not config.inference:
            self.optimizer_critic = optim.Adam(self.critic_dnn.parameters(), lr=config.lr)
            self.lr_scheduler_critic = optim.lr_scheduler.ExponentialLR(
                self.optimizer_critic, 0.9, verbose=True
            )
            self.critic_loss_fn = nn.MSELoss()
        self.critic_dnn.to(self.device)

    
    def reinforce_step(self, states_batch, states_len, len_mask):

        states_batch_dev = torch.tensor(states_batch, dtype=torch.float32).unsqueeze(-1).to(self.device)
        len_mask = torch.as_tensor(len_mask)
        len_mask_device = len_mask.to(self.device) # Keep one in device and other in cpu 
        # to minimize cpu-gpu exchanges (since len_mask is used in cpu-only functions as well) 

        # Compute actions (i.e. sequence in which items are allocated to bins)
        self.optimizer_actor.zero_grad()
        log_probs_actions, actions = self.policy_dnn(states_batch_dev, states_len, len_mask, len_mask_device)
        log_prob_seq = torch.sum(
            log_probs_actions, dim=1
        )  # Summing log_probs is equivalent to multiplying probs
        # log_prob_seq is now the log prob of having taken action A under policy pi. i.e. pi(At|St).
        
        # Compute predicted reward and backpropagate loss
        self.optimizer_critic.zero_grad()
        pred_reward = self.critic_dnn(states_batch_dev, states_len, len_mask_device)
        real_reward = compute_reward(self.config, states_batch, len_mask, actions)
        real_reward = torch.tensor(real_reward, dtype=torch.float32, requires_grad=False).to(self.device)
        critic_loss = self.critic_loss_fn(
            real_reward, pred_reward
        )
        critic_loss.backward()
        self.optimizer_critic.step()

        # Compute agent reward and backpropagate loss
        pred_reward = pred_reward.detach()
        reward_gap = (real_reward - pred_reward) * 100.0
        agent_loss = torch.mean(log_prob_seq * reward_gap * -1.0)
        agent_loss.backward()
        self.optimizer_actor.step()

        return real_reward.mean().cpu(), pred_reward.cpu().numpy().mean()

    
    def apply_policy(self, states_batch, states_len, len_mask):
        
        states_batch_dev = torch.tensor(states_batch, dtype=torch.float32).unsqueeze(-1).to(self.device)
        len_mask = torch.as_tensor(len_mask)
        len_mask_device = len_mask.to(self.device) # Keep one in device and other in cpu 
        # to minimize cpu-gpu exchanges (since len_mask is used in cpu-only functions as well) 

        # Compute actions (i.e. sequence in which items are allocated to bins)
        actions = self.policy_dnn.inference(states_batch_dev, states_len, len_mask, len_mask_device)

        return actions.type(torch.int32)