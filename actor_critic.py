from symbol import parameters
from tabnanny import verbose
import torch
import torch.optim as optim
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
import numpy as np
from rl_env import avg_occupancy, compute_reward
from dnns import ActorPointerNetwork, CriticNetwork
from argparse import Namespace


class Actor:
    def __init__(self, config: Namespace):

        self.batch_size = config.batch_size
        self.max_len = config.max_num_items
        self.policy_dnn = ActorPointerNetwork(config)
        self.critic_dnn = CriticNetwork(config)
        self.optimizer_actor = optim.Adam(self.policy_dnn.parameters(), lr=config.lr)
        self.optimizer_critic = optim.Adam(self.critic_dnn.parameters(), lr=config.lr)
        self.lr_scheduler_actor = optim.lr_scheduler.ExponentialLR(
            self.optimizer_actor, 0.9, verbose=True
        )
        self.lr_scheduler_critic = optim.lr_scheduler.ExponentialLR(
            self.optimizer_critic, 0.9, verbose=True
        )
        self.critic_loss_fn = nn.MSELoss()
        self.config = config
        
        self.device = config.device if torch.cuda.is_available() else "cpu"
        self.policy_dnn.to(self.device)
        self.critic_dnn.to(self.device)

    def reinforce_step(self, states_batch, states_len, len_mask, verbose=False):

        states_batch = torch.tensor(states_batch, dtype=torch.float32).unsqueeze(-1).to(self.device)
        len_mask = torch.tensor(len_mask, dtype=torch.int32).to(self.device)

        self.optimizer_actor.zero_grad()
        log_probs_actions, actions = self.policy_dnn(states_batch, states_len, len_mask)
        log_prob_seq = torch.sum(
            log_probs_actions, dim=1
        )  # Equivalent to multiplying probs
        # log_prob_seq is now the log prob of having taken action A under policy pi. i.e. pi(At|St).
        
        self.optimizer_critic.zero_grad()
        pred_reward = self.critic_dnn(states_batch, states_len, len_mask)

        real_reward = compute_reward(self.config, states_batch.detach().cpu(), len_mask.detach().cpu(), actions.detach().cpu())
        real_reward = torch.tensor(real_reward, dtype=torch.float32, requires_grad=False).to(self.device)
        critic_loss = self.critic_loss_fn(
            real_reward, pred_reward
        )
        critic_loss.backward()
        self.optimizer_critic.step()
        pred_reward = pred_reward.detach()

        if verbose:
            print("Critic gradients")
            for name, param in self.critic_dnn.named_parameters():
                if param.grad is None:
                    print(f" - Layer: {name} has no gradient")
                else:
                    print(
                        f" - Layer: {name}, Max Grad: {param.grad.max():.1e}, Min Grad: {param.grad.min():.1e}"
                    )
            print("\n")

        reward_gap = (real_reward - pred_reward) * 100.0
        agent_loss = torch.mean(log_prob_seq * reward_gap * -1.0)
        agent_loss.backward()
        self.optimizer_actor.step()

        if verbose:
            print("Actor gradients")
            for name, param in self.policy_dnn.named_parameters():
                if param.grad is None:
                    print(f" - Layer: {name} has no gradient")
                else:
                    print(
                        f" - Layer: {name}, Max Grad: {param.grad.max():.1e}, Min Grad: {param.grad.min():.1e}"
                    )
            print("\n")

        return (
            agent_loss.item(),
            critic_loss.item(),
            real_reward.mean().cpu(),
            pred_reward.cpu().numpy().mean(),
        )
