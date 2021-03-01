#!/usr/bin/env python3

import torch
import cherry
import gym
import numpy as np
from itertools import count
import statistics

NUM_ENVS = 6
STEPS = 5
TRAIN_STEPS = int(1e4)

class A2C(torch.nn.Module):
    def __init__(self, num_envs):
        super(A2C, self).__init__()
        
        self.num_envs = num_envs
        self.gamma = 0.99
        self.vf_coef = 0.25
        self.ent_coef = 0.01
        self.max_clip_norm = 0.5

    def select_action(self, state):
        probs, value = self(state)
        mass = torch.distributions.Categorical(probs)
        action = mass.sample()
        # Return selected action, logprob, value estimation and categorical entropy
        return action, {"log_prob": mass.log_prob(action), "value": value, "entropy": mass.entropy()}

    
    def learn_step(self, replay, optimizer):
        policy_loss = []
        value_loss = []
        entropy_loss = []

        # Discount rewards and boostrap them with the estimation from the next state
        last_action, last_value = self(replay.next_state()[-1,:,:])
        # Boostrap from zero if it is a terminal state
        last_value = (last_value[:, 0]*(1 - replay.done()[-1]))

        rewards = cherry.td.discount(self.gamma, replay.reward(), replay.done(), last_value)
        for sars, reward in zip(replay, rewards):
            log_prob = sars.log_prob.view(self.num_envs, -1)
            value = sars.value.view(self.num_envs, -1)
            entropy = sars.entropy.view(self.num_envs, -1)
            reward = reward.view(self.num_envs, -1)

            # Compute advantage
            advantage = reward - value
            
            # Compute policy gradient loss
            # (advantage.detach() because you do not have to backward on the advantage path) 
            policy_loss.append(-log_prob * advantage.detach())
            # Compute value estimation loss
            value_loss.append((reward - value)**2)
            # Compute entropy loss
            entropy_loss.append(entropy)
        

        # Compute means over accumulated errors
        value_loss = torch.stack(value_loss).mean()
        policy_loss = torch.stack(policy_loss).mean()
        entropy_loss = torch.stack(entropy_loss).mean()

        # Take an optimization step
        optimizer.zero_grad()
        loss = policy_loss + self.vf_coef * value_loss - self.ent_coef * entropy_loss
        loss.backward()
        # Clip gradients
        torch.nn.utils.clip_grad_norm_(self.parameters(), self.max_clip_norm)
        optimizer.step()




class A2CPolicy(A2C):
    def __init__(self, state_size, action_size, num_envs):
        super(A2CPolicy, self).__init__(num_envs)
        self.state_size = state_size
        self.action_size = action_size
        self.n_hidden = 128

        # Backbone net
        self.net = torch.nn.Sequential(
            torch.nn.Linear(self.state_size, self.n_hidden),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(self.n_hidden, self.n_hidden),
            torch.nn.LeakyReLU(),
        )

        # Action head (policy gradient)
        self.action_head = torch.nn.Sequential(
            torch.nn.Linear(self.n_hidden, self.action_size),
            torch.nn.Softmax(dim=1)
        )

        # Value estimation head (A2C)
        self.value_head = torch.nn.Sequential(
            torch.nn.Linear(self.n_hidden, 1),
        )


    def forward(self, x):
        # Return both the action probabilities and the value estimations
        return self.action_head(self.net(x)), self.value_head(self.net(x))

if __name__ == '__main__':
    env = gym.vector.make('CartPole-v0', num_envs=NUM_ENVS)
    env = cherry.envs.Logger(env, interval=1000)
    env = cherry.envs.Torch(env)

    policy = A2CPolicy(env.state_size, env.action_size, NUM_ENVS)
    optimizer = torch.optim.RMSprop(policy.parameters(), lr=7e-4, eps=1e-5, alpha=0.99)
    
    state = env.reset()
    for train_step in range(0, TRAIN_STEPS):
        replay = cherry.ExperienceReplay()
        for step in range(0, STEPS):
            action, info = policy.select_action(state)
            new_state, reward, done, _ = env.step(action)
            replay.append(state, action, reward, new_state, done, **info)
            state = new_state
            
        policy.learn_step(replay, optimizer)
    
    env = gym.make('CartPole-v0')
    env = cherry.envs.Torch(env)
    env = cherry.envs.Runner(env)    
    while True:
        env.run(lambda state: policy.select_action(state), episodes=1, render=True)
