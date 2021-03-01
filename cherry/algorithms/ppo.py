#!/usr/bin/env python3

"""
**Description**

Helper functions for implementing PPO.
"""

import torch as th
from cherry import debug


def policy_loss(new_log_probs, old_log_probs, advantages, clip=0.1, dual_clip=None):
    """
    [[Source]](https://github.com/seba-1511/cherry/blob/master/cherry/algorithms/ppo.py)

    **Description**

    The dual clipped policy loss of Dual-Clip Proximal Policy Optimization.

    **References**

    1. Deheng Ye et al. 2020 . “ Mastering Complex Control in MOBA Games with Deep Reinforcement Learning.” arXiv:1912.09729 .

    **Arguments**

    * **new_log_probs** (tensor) - The log-density of actions from the target policy.
    * **old_log_probs** (tensor) - The log-density of actions from the behaviour policy.
    * **advantages** (tensor) - Advantage of the actions.
    * **clip** (float, *optional*, default=0.1) - The clipping coefficient.
    * **dual_clip** (float, *optional*, default=None) - The dual-clipping coefficient.

    **Returns**

    * (tensor) - The dual-clipped policy loss for the given arguments.

    **Example**

    ~~~python
    advantage = ch.pg.generalized_advantage(GAMMA,
                                            TAU,
                                            replay.reward(),
                                            replay.done(),
                                            replay.value(),
                                            next_state_value)
    new_densities = policy(replay.state())
    new_logprobs = new_densities.log_prob(replay.action())
    loss = loss_dual_clip(new_logprobs,
                       replay.logprob().detach(),
                       advantage.detach(),
                       clip=0.2,
                       dual_clip=2)
    ~~~
    """
    msg = "new_log_probs, old_log_probs and advantages must have equal size."
    assert new_log_probs.size() == old_log_probs.size() == advantages.size(), msg
    if debug.IS_DEBUGGING:
        if old_log_probs.requires_grad:
            debug.logger.warning(
                "PPO:policy_loss: old_log_probs.requires_grad is True."
            )
        if advantages.requires_grad:
            debug.logger.warning("PPO:policy_loss: advantages.requires_grad is True.")
        if not new_log_probs.requires_grad:
            debug.logger.warning(
                "PPO:policy_loss: new_log_probs.requires_grad is False."
            )
    ratios = th.exp(new_log_probs - old_log_probs)
    obj = ratios * advantages
    obj_clip = ratios.clamp(1.0 - clip, 1.0 + clip) * advantages
    if dual_clip is not None:
        obj_dual_clip = dual_clip * advantages

        return -(
            (th.max(th.min(obj, obj_clip), obj_dual_clip)[advantages < 0]).mean()
            + (th.min(obj, obj_clip)[advantages > 0]).mean()
        )

    return -th.min(obj, obj_clip).mean()


def state_value_loss(new_values, old_values, rewards, clip=0.1):
    """
    [[Source]](https://github.com/seba-1511/cherry/blob/master/cherry/algorithms/ppo.py)

    **Description**

    The clipped state-value loss of Proximal Policy Optimization.

    **References**

    1. PPO paper

    **Arguments**

    * **new_values** (tensor) - State values from the optimized value function.
    * **old_values** (tensor) - State values from the reference value function.
    * **rewards** (tensor) -  Observed rewards.
    * **clip** (float, *optional*, default=0.1) - The clipping coefficient.

    **Returns**

    * (tensor) - The clipped value loss for the given arguments.

    **Example**

    ~~~python
    values = v_function(batch.state())
    value_loss = ppo.state_value_loss(values,
                                      batch.value().detach(),
                                      batch.reward(),
                                      clip=0.2)
    ~~~
    """
    msg = "new_values, old_values, and rewards must have equal size."
    assert new_values.size() == old_values.size() == rewards.size(), msg
    if debug.IS_DEBUGGING:
        if old_values.requires_grad:
            debug.logger.warning(
                "PPO:state_value_loss: old_values.requires_grad is True."
            )
        if rewards.requires_grad:
            debug.logger.warning("PPO:state_value_loss: rewards.requires_grad is True.")
        if not new_values.requires_grad:
            debug.logger.warning(
                "PPO:state_value_loss: new_values.requires_grad is False."
            )
    loss = (rewards - new_values) ** 2
    clipped_values = old_values + (new_values - old_values).clamp(-clip, clip)
    clipped_loss = (rewards - clipped_values) ** 2
    return 0.5 * th.max(loss, clipped_loss).mean()
