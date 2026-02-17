import os
from collections import defaultdict
from matplotlib import pyplot as plt
import torch
from torch import nn
import torchrl
import torchrl.envs as torch_envs
from tqdm import tqdm
import gymnasium as gym
import tensordict
from tensordict import nn as dict_nn
import torchsummary
import torchvision
from torchrl.modules.tensordict_module import ProbabilisticActor
from torchrl.modules import ValueOperator
from torchrl.objectives.value import GAE
from torchrl.objectives import ClipPPOLoss
from torchrl.collectors import SyncDataCollector
from torchrl.data import ReplayBuffer, LazyTensorStorage, SamplerWithoutReplacement
from tensordict.nn import set_interaction_type, InteractionType
import pandas as pd
import numpy as np
from math import sqrt, log
from collections import deque
from warship_env import WarshipEnv


class PPO:
    def __init__(self, action_spec, observation_spec, lr=3e-4, max_grad_norm=1.0, clip_epsilon=0.2, gamma=0.99, lmbda=0.95, entropy_eps=1e-4):
        self.device = torch.device('cuda')
        self.max_grad_norm = max_grad_norm
        self.lr = lr
        self.clip_epsilon = clip_epsilon
        self.gamma = gamma
        self.lmbda = lmbda
        self.entropy_eps = entropy_eps
        self.build_models(action_spec, observation_spec)

    def build_models(self, action_spec, observation_spec):
        '''builds actor, critic, and loss modules'''
        input_shape = observation_spec['observation'].shape
        output_shape = action_spec.shape
        self.actor = nn.Sequential(
                        nn.Linear(input_shape[-1], 128, device=self.device),
                        nn.Tanh(),
                        # nn.Linear(128, 128, device=self.device),
                        # nn.Tanh(),
                        nn.Linear(128, 128, device=self.device),
                        nn.Tanh(),
                        nn.Linear(128, 128, device=self.device),
                        nn.Tanh(),
                        nn.Linear(128, output_shape[-1], device=self.device),
        )
        
        self.policy_module = dict_nn.TensorDictModule(self.actor, in_keys=["observation"], out_keys=["logits"])
        self.policy_module = ProbabilisticActor(module=self.policy_module,
                                            spec=action_spec,
                                            in_keys=["logits"],
                                            distribution_class=torch.distributions.OneHotCategorical,
                                            return_log_prob=True)
        
        self.value_net = nn.Sequential(
            nn.Linear(input_shape[-1], 24, device=self.device),
            nn.Tanh(),
            nn.Linear(24, 64, device=self.device),
            nn.Tanh(),
            # nn.Linear(64, 64, device=self.device),
            # nn.Tanh(),
            nn.Linear(64, 64, device=self.device),
            nn.Tanh(),
            nn.Linear(64, 1, device=self.device),
        )
        
        self.value_module = ValueOperator(self.value_net, in_keys=["observation"])
        
        self.advantage_module = GAE(gamma=self.gamma, lmbda=self.lmbda, value_network=self.value_module, average_gae=True, device=self.device)
        self.loss_module = ClipPPOLoss(actor_network=self.policy_module, critic_network=self.value_module, clip_epsilon=self.clip_epsilon, entropy_coeff=self.entropy_eps)

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            "actor": self.actor.state_dict(),
            "value_net": self.value_net.state_dict(),
        }, path)

class PPOManager:
    def __init__(self, ppo: PPO, env, logged_env, lr=3e-4, save_dir="checkpoints", save_interval=10):
        self.ppo = ppo
        self.env = env
        self.logged_env = logged_env
        self.optimizer = torch.optim.Adam(self.ppo.loss_module.parameters(), lr=lr)
        self.save_dir = save_dir
        self.save_interval = save_interval

    def train(self, frames_per_batch, total_frames, sub_batch_size, num_epochs):
        collector = SyncDataCollector(self.env, self.ppo.policy_module, frames_per_batch=frames_per_batch, total_frames=total_frames, device=self.ppo.device)
        replay_buffer = ReplayBuffer(storage=LazyTensorStorage(max_size=frames_per_batch), sampler=SamplerWithoutReplacement())
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, total_frames // frames_per_batch, 0.0)
        
        logs = defaultdict(list)
        pbar = tqdm(total=total_frames)
        eval_str = ""

        for i, tensordict_data in enumerate(collector):
            for _ in range(num_epochs):
                self.ppo.advantage_module(tensordict_data)
                data_view = tensordict_data.reshape(-1)
                replay_buffer.extend(data_view.cpu())
                for _ in range(frames_per_batch // sub_batch_size):
                    subdata = replay_buffer.sample(sub_batch_size)
                    loss_vals = self.ppo.loss_module(subdata.to(self.ppo.device))
                    loss_value = loss_vals["loss_objective"] + loss_vals["loss_critic"] + loss_vals["loss_entropy"]
                    loss_value.backward()
                    torch.nn.utils.clip_grad_norm_(self.ppo.loss_module.parameters(), self.ppo.max_grad_norm)
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    
                logs["reward"].append(tensordict_data["next", "reward"].mean().item())
                pbar.update(tensordict_data.numel())
                
            cum_reward_str = (
                f"average reward={logs['reward'][-1]: 4.4f} (init={logs['reward'][0]: 4.4f})"
            )
            # logs["step_count"].append(tensordict_data["step_count"].max().item())
            # stepcount_str = f"step count (max): {logs['step_count'][-1]}"
            logs["lr"].append(self.optimizer.param_groups[0]["lr"])
            lr_str = f"lr policy: {logs['lr'][-1]: 4.4f}"
            
            if i % 3 == 0:
                with set_interaction_type(InteractionType.DETERMINISTIC), torch.no_grad():  # magic
                    eval_rollout = self.logged_env.rollout(600, self.ppo.policy_module)
                    self.logged_env.transform[-1].dump()
                    logs["eval reward"].append(eval_rollout["next", "reward"].mean().item())
                    logs["eval reward (sum)"].append(
                        eval_rollout["next", "reward"].sum().item()
                    )
                    # logs["eval step_count"].append(eval_rollout["step_count"].max().item())
                    eval_str = (
                        f"eval cumulative reward: {logs['eval reward (sum)'][-1]: 4.4f} "
                        f"(init: {logs['eval reward (sum)'][0]: 4.4f}), "
                        # f"eval step-count: {logs['eval step_count'][-1]}"
                    )
                    del eval_rollout

            if i % self.save_interval == 0:
                self.ppo.save(os.path.join(self.save_dir, f"checkpoint_{i}.pt"))

            # pbar.set_description(", ".join([eval_str, cum_reward_str, stepcount_str, lr_str]))
            scheduler.step()

class CompetitivePPOManager:
    def __init__(self, ppo: PPO, env, logged_env, lr=3e-4, exploration_constant=0.3, win_rate_goal=0.8, results_len=100, save_dir="checkpoints", save_interval=10):
        self.ppo = ppo
        self.env = env
        self.logged_env = logged_env
        self.optimizer = torch.optim.Adam(self.ppo.loss_module.parameters(), lr=lr)
        self.save_dir = save_dir
        self.save_interval = save_interval
        self.models_df = pd.DataFrame(columns=["iteration", "results", "policy", "collector"])
        self.exploration_constant = exploration_constant
        self.total_iteration_visits = 0
        self.win_rate_goal = win_rate_goal
        self.results_len = results_len
        
    
    def utc_ranking(self, row: pd.Series):
        # monte carlo search uses UTC = average_reward + C * sqrt(ln(parent_visits) / child_visits)
        # in this case we do not have a tree structure, so parent visits is the total visits
        # average reward might be wins / losses, the win rate for the given model in the current iteration
        # C should probably be low, exploitation is much more important than exploration
        
        if len(row['results']) == 0:
            return float('inf')
        return row['win_rate'] + self.exploration_constant * sqrt(log(self.total_iteration_visits) / len(row['results']))
        
    
    def select_opponent(self):
        return self.models_df[np.argmax(self.models_df.apply(self.utc_ranking))]
         
    def train(self, frames_per_batch, total_frames, sub_batch_size, num_epochs):
        replay_buffer = ReplayBuffer(storage=LazyTensorStorage(max_size=frames_per_batch), sampler=SamplerWithoutReplacement())
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, total_frames // frames_per_batch, 0.0)

        iteration = -1
        i = -1
        while True:
            iterations += 1
            self.env.opponent_policy = self.ppo.policy_module
            collector = SyncDataCollector(self.env, self.ppo.policy_module, frames_per_batch=frames_per_batch, total_frames=total_frames, device=self.ppo.device)
            self.models_df = pd.concat(self.model_df, {"iteration" : iteration, "results" : deque([]), "policy" : self.ppo.policy_module, "collector" : collector})
            
            while min(self.models_df['win_rate']) > (1 - self.win_rate_goal):
                i += 1
                opp_data = self.select_opponent()
                collector = opp_data['collector']
                tensordict_data = next(collector)
                for _ in range(num_epochs):
                    self.ppo.advantage_module(tensordict_data)
                    data_view = tensordict_data.reshape(-1)
                    replay_buffer.extend(data_view)
                    for _ in range(frames_per_batch // sub_batch_size):
                        subdata = replay_buffer.sample(sub_batch_size)
                        loss_vals = self.ppo.loss_module(subdata.to(self.ppo.device))
                        total_loss = loss_vals["loss_objective"] + loss_vals["loss_critic"] + loss_vals["loss_entropy"]
                        total_loss.backward()
                        torch.nn.utils.clip_grad_norm_(self.ppo.loss_module.parameters(), self.ppo.max_grad_norm)
                        self.optimizer.step()
                        self.optimizer.zero_grad()
                        
                if i % 3 == 0:
                    opp_data = self.select_opponent()
                    self.logged_env.opponent_policy = opp_data['policy']
                    with set_interaction_type(InteractionType.DETERMINISTIC), torch.no_grad():
                        eval_rollout = self.logged_env.rollout(600, self.ppo.policy_module)
                        self.logged_env.transform[-1].dump()

                if i % self.save_interval == 0:
                    self.ppo.save(os.path.join(self.save_dir, f"checkpoint_{i}.pt"))