import random
import gym
import numpy as np
import pybullet_envs

from itertools import count

import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import cherry as ch
import cherry.policies as policies
import cherry.models as models
import cherry.envs as envs
from cherry.rewards import discount_rewards

RECORD = True
SEED = 567
TOTAL_STEPS = 1000000
GAMMA = 0.99
V_WEIGHT = 0.5

random.seed(SEED)
np.random.seed(SEED)
th.manual_seed(SEED)


class ActorCriticNet(nn.Module):
    def __init__(self, env):
        super(ActorCriticNet, self).__init__()
        self.actor = models.control.Actor(env.state_size,
                                          env.action_size,
                                          layer_sizes=[64, 64])
        self.critic = models.control.ControlMLP(env.state_size, 1)

        self.action_dist = policies.ActionDistribution(env,
                                                       use_probs=False,
                                                       reparam=False)

    def forward(self, x):
        action_scores = self.actor(x)
        action_density = self.action_dist(action_scores)
        value = self.critic(x)
        return action_density, value


def update(replay, optimizer):
    policy_loss = []
    value_loss = []

    # Discount and normalize rewards
    rewards = discount_rewards(GAMMA, replay.rewards, replay.dones)
    rewards = ch.utils.normalize(ch.utils.totensor(rewards))

    # Compute losses
    for info, reward in zip(replay.infos, rewards[0]):
        log_prob = info['log_prob']
        value = info['value']
        policy_loss.append(-log_prob * (reward - value.item()))
        value_loss.append(F.mse_loss(value, reward.detach()))  # Don't perform gradient of reward

    # Optimize network
    optimizer.zero_grad()
    loss = th.stack(policy_loss).sum() + V_WEIGHT * th.stack(value_loss).sum()
    loss.backward(retain_graph=True)
    optimizer.step()


def get_action_value(state, policy):
    dist, value = policy(state)
    action = dist.sample()
    info = {
        'log_prob': dist.log_prob(action),
        'value': value
    }
    return action, info

if __name__ == '__main__':
    env_name = 'AntBulletEnv-v0'
    env = gym.make(env_name)
    env = envs.Logger(env)
    env = envs.OpenAINormalize(env)
    env = envs.Torch(env)
    env = envs.Runner(env)
    env.seed(SEED)

    if RECORD:
        record_env = gym.make(env_name)
        record_env = envs.Monitor(record_env, './videos/')
        record_env = envs.OpenAINormalize(record_env)
        record_env = envs.Torch(record_env)
        record_env = envs.Runner(record_env)
        record_env.seed(SEED)

    policy = ActorCriticNet(env)
    optimizer = optim.Adam(policy.parameters(), lr=1e-2)
    replay = ch.ExperienceReplay()
    get_action = lambda state: get_action_value(state, policy)

    total_steps = 0
    for episode in count(1):
        num_samples, num_episodes = env.run(get_action, replay, episodes=1)
        rewards = replay.rewards

        # Update policy
        update(replay, optimizer)
        replay.empty()

        # Record env every 10 episodes
        if RECORD and episode % 10 == 0:
            record_env.run(get_action, episodes=3, render=True)

        # Termination Condition: reaching max number of timesteps
        total_steps += num_samples
        if (total_steps >= TOTAL_STEPS):
            print('Reached maximum number of timesteps. Avg Reward: {}'.format(np.average(rewards)))
            break
