#!/usr/bin/env python3

"""
**Description**

A set of common distributions.

"""

import torch as th
import torch.nn as nn
import cherry as ch

from torch.distributions import (Categorical,
                                 Normal,
                                 Distribution)

from gym.spaces import Discrete


class Reparameterization(object):

    """
    [[Source]](https://github.com/seba-1511/cherry/blob/master/cherry/distributions.py)

    **Description**

    Unifies interface for distributions that support `rsample` and those that do not.

    When calling `sample()`, this class checks whether `density` has a `rsample()` member,
    and defaults to call `sample()` if it does not.

    **References**

    1. Kingma and Welling. 2013. “Auto-Encoding Variational Bayes.” arXiv [stat.ML].

    **Arguments**

    * **density** (Distribution) - The distribution to wrap.

    **Example**

    ~~~python
    density = Normal(mean, std)
    reparam = Reparameterization(density)
    sample = reparam.sample()  # Uses Normal.rsample()
    ~~~

    """

    def __init__(self, density):
        self.density = density

    def sample(self, *args, **kwargs):
        if self.density.has_rsample:
            return self.density.rsample(*args, **kwargs)
        return self.density.sample(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self.density, name)

    def __repr__(self):
        return str(self)

    def __str__(self):
        return 'Reparameterization(' + str(self.density) + ')'


class ActionDistribution(nn.Module):
    """
    [[Source]](https://github.com/seba-1511/cherry/blob/master/cherry/distributions.py)

    **Description**

    A helper module to automatically choose the proper policy distribution,
    based on the Gym environment `action_space`.

    For `Discrete` action spaces, it uses a `Categorical` distribution, otherwise
    it uses a `Normal` which uses a diagonal covariance matrix.

    This class enables to write single version policy body that will be compatible
    with a variety of environments.

    **Arguments**

    * **env** (Environment) - Gym environment for which actions will be sampled.
    * **logstd** (float/tensor, *optional*, default=0) - The log standard
    deviation for the `Normal` distribution.
    * **use_probs** (bool, *optional*, default=False) - Whether to use probabilities or logits
    for the `Categorical` case.
    * **reparam** (bool, *optional*, default=False) - Whether to use reparameterization in the
    `Normal` case.

    **Example**

    ~~~python
    env = gym.make('CartPole-v1')
    action_dist = ActionDistribution(env)
    ~~~

    """

    def __init__(self, env, logstd=None, use_probs=False, reparam=False):
        super(ActionDistribution, self).__init__()
        self.use_probs = use_probs
        self.reparam = reparam
        self.is_discrete = ch.envs.is_discrete(env.action_space)
        if not self.is_discrete:
            if logstd is None:
                action_size = ch.envs.get_space_dimension(env.action_space,
                                                          vectorized_dims=False)
                logstd = nn.Parameter(th.zeros(action_size))
            if isinstance(logstd, (float, int)):
                logstd = nn.Parameter(th.Tensor([logstd]))
            self.logstd = logstd

    def forward(self, x):
        if self.is_discrete:
            if self.use_probs:
                return Categorical(probs=x)
            return Categorical(logits=x)
        else:
            density = Normal(loc=x, scale=self.logstd.exp())
            if self.reparam:
                density = Reparameterization(density)
            return density


class TanhNormal(Distribution):

    """
    [[Source]](https://github.com/seba-1511/cherry/blob/master/cherry/models/tabular.py)

    **Description**

    Implements a Normal distribution followed by a Tanh, often used with the Soft Actor-Critic.

    This implementation also exposes `sample_and_log_prob` and `rsample_and_log_prob`,
    which returns both samples and log-densities.
    The log-densities are computed using the pre-activation values for numerical stability.

    **Credit**

    Adapted from Vitchyr Pong's RLkit:
    https://github.com/vitchyr/rlkit/blob/master/rlkit/torch/distributions.py

    **References**

    1. Haarnoja et al. 2018. “Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor.” arXiv [cs.LG].
    2. Haarnoja et al. 2018. “Soft Actor-Critic Algorithms and Applications.” arXiv [cs.LG].

    **Arguments**

    * **normal_mean** (tensor) - Mean of the Normal distribution.
    * **normal_std** (tensor) - Standard deviation of the Normal distribution.

    **Example**
    ~~~python
    mean = th.zeros(5)
    std = th.ones(5)
    dist = TanhNormal(mean, std)
    samples = dist.rsample()
    logprobs = dist.log_prob(samples)  # Numerically unstable :(
    samples, logprobs = dist.rsample_and_log_prob()  # Stable :)
    ~~~

    """

    def __init__(self, normal_mean, normal_std):
        self.normal_mean = normal_mean
        self.normal_std = normal_std
        self.normal = Normal(normal_mean, normal_std)

    def sample_n(self, n):
        z = self.normal.sample_n(n)
        return th.tanh(z)

    def log_prob(self, value):
        pre_tanh_value = (th.log1p(value) - th.log1p(-value)).mul(0.5)
        offset = th.log1p(-value**2 + 1e-6)
        return self.normal.log_prob(pre_tanh_value) - offset

    def sample(self):
        z = self.normal.sample().detach()
        return th.tanh(z)

    def mean(self):
        return th.tanh(self.normal.mean)

    def mode(self):
        return self.mean()

    def sample_and_log_prob(self):
        z = self.normal.sample().detach()
        value = th.tanh(z)
        offset = th.log1p(-value**2 + 1e-6)
        log_prob = self.normal.log_prob(z) - offset
        return value, log_prob

    def rsample_and_log_prob(self):
        z = self.normal.rsample()
        value = th.tanh(z)
        offset = th.log1p(-value**2 + 1e-6)
        log_prob = self.normal.log_prob(z) - offset
        return value, log_prob

    def rsample(self):
        z = self.normal.rsample()
        z.requires_grad_()
        return th.tanh(z)
