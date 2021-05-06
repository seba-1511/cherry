#!/usr/bin/env python3

import numpy as np
from .base import Wrapper
from .utils import num_envs


class RewardNormalizer(Wrapper):

    """
    [[Source]](https://github.com/seba-1511/cherry/blob/master/cherry/envs/normalizer_wrapper.py)

    **Description**

    Normalizes the rewards with a running average.

    **Arguments**

     * **env** (Environment) - Environment to normalize.
     * **statistics** (dict, *optional*, default=None) - Dictionary used to
        bootstrap the normalizing statistics.
     * **beta** (float, *optional*, default=0.99) - Moving average weigth.
     * **eps** (float, *optional*, default=1e-8) - Numerical stability.

    **Credit**

    Adapted from Tristan Deleu's implementation.

    **Example**
    ~~~python
    env = gym.make('CartPole-v0')
    env = cherry.envs.RewardNormalizer(env)
    env2 = gym.make('CartPole-v0')
    env2 = cherry.envs.RewardNormalizer(env2,
                                       statistics=env.statistics)
    ~~~
    """

    def __init__(self, env, statistics=None, beta=0.99, eps=1e-8):
        super(RewardNormalizer, self).__init__(env)
        self.beta = beta
        self.eps = eps
        nenvs = num_envs(self.env)
        if statistics is not None and 'mean' in statistics:
            self._reward_mean = np.copy(statistics['mean'])
        else:
            self._reward_mean = np.zeros(nenvs)

        if statistics is not None and 'var' in statistics:
            self._reward_var = np.copy(statistics['var'])
        else:
            self._reward_var = np.ones(nenvs)

    @property
    def statistics(self):
        return {
            'mean': self._reward_mean,
            'var': self._reward_var,
        }

    def _reward_normalize(self, reward):
        self._reward_mean = self.beta * self._reward_mean + (1.0 - self.beta) * reward
        self._reward_var = self.beta * self._reward_var + (1.0 - self.beta) * np.square(reward - self._reward_mean)
        return (reward - self._reward_mean) / (np.sqrt(self._reward_var) + self.eps)

    def step(self, *args, **kwargs):
        state, reward, done, infos = self.env.step(*args, **kwargs)
        return state, self._reward_normalize(reward), done, infos
