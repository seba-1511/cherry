#!/usr/bin/env python3

"""
**Description**

Utilities to implement policy gradient algorithms.
"""

import torch as th
import cherry as ch

from cherry._utils import _reshape_helper


def generalized_advantage(gamma,
                          tau,
                          rewards,
                          dones,
                          values,
                          next_value):
    """
    **Description**

    Computes the generalized advantage estimator. (GAE)

    **References**

    1. Schulman et al. 2015. “High-Dimensional Continuous Control Using Generalized Advantage Estimation”
    2. https://github.com/joschu/modular_rl/blob/master/modular_rl/core.py#L49

    **Arguments**

    * **gamma** (float) - Discount factor.
    * **tau** (float) - Bias-variance trade-off.
    * **rewards** (tensor) - Tensor of rewards.
    * **dones** (tensor) - Tensor indicating episode termination.
      Entry is 1 if the transition led to a terminal (absorbing) state, 0 else.
    * **values** (tensor) - Values for the states producing the rewards.
    * **next_value** (tensor) - Value of the state obtained after the
      transition from the state used to compute the last value in `values`.

    **Returns**

    * tensor - Tensor of advantages.

    **Example**
    ~~~python
    mass, next_value = policy(replay[-1].next_state)
    advantages = generalized_advantage(0.99,
                                       0.95,
                                       replay.reward(),
                                       replay.value(),
                                       replay.done(),
                                       next_value)
    ~~~
    """

    rewards = _reshape_helper(rewards)
    dones = _reshape_helper(dones)
    values = _reshape_helper(values)
    next_value = _reshape_helper(next_value)
    next_value = ch.totensor(next_value).reshape_as(values[0].unsqueeze(0))

    msg = 'rewards, values, and dones must have equal length.'
    assert len(values) == len(rewards) == len(dones), msg

    next_values = th.cat((values[1:], next_value), dim=0)
    td_errors = ch.temporal_difference(gamma, rewards, dones, values, next_values)
    advantages = ch.discount(tau * gamma, td_errors, dones)
    return advantages
