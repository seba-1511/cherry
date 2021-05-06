#!/usr/bin/env python3

import unittest
import torch as th

from cherry.algorithms import ppo

BSZ = 10


def ref_policy_loss(new_log_probs, old_log_probs, advantages, clip=0.1):
    # Compute clipped policy loss
    loss = 0.0
    for i in range(len(new_log_probs)):
        ratio = th.exp(new_log_probs[i] - old_log_probs[i])
        objective = ratio * advantages[i]
        objective_clipped = ratio.clamp(1.0 - clip,
                                        1.0 + clip) * advantages[i]
        policy_loss = - th.min(objective, objective_clipped)
        loss += policy_loss
    return loss / len(new_log_probs)


def ref_value_loss(new_values, old_values, rewards, clip=0.1):
    # Compute clipped value loss
    loss = 0.0
    for i in range(len(new_values)):
        value_loss = (rewards[i] - new_values[i])**2
        old_value = old_values[i]
        clipped_value = old_value + (new_values[i] - old_value).clamp(-clip, clip)
        clipped_loss = (rewards[i] - clipped_value)**2
        value_loss = 0.5 * th.max(value_loss, clipped_loss)
        loss += value_loss
    return loss / len(new_values)


class TestPPOAlgorithms(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_ppo_policy_loss(self):
        for _ in range(10):
            for shape in [(1, BSZ), (BSZ, 1), (BSZ, )]:
                for clip in [0.0, 0.1, 0.2, 1.0]:
                    new_log_probs = th.randn(BSZ)
                    old_log_probs = th.randn(BSZ)
                    advantages = th.randn(BSZ)
                    ref = ref_policy_loss(new_log_probs,
                                          old_log_probs,
                                          advantages,
                                          clip=clip)
                    loss = ppo.policy_loss(new_log_probs.view(*shape),
                                           old_log_probs.view(*shape),
                                           advantages.view(*shape),
                                           clip=clip)
                    self.assertAlmostEqual(loss.item(), ref.item(), places=6)

    def test_ppo_value_loss(self):
        for _ in range(10):
            for shape in [(1, BSZ), (BSZ, 1), (BSZ, )]:
                for clip in [0.0, 0.1, 0.2, 1.0]:
                    new_values = th.randn(BSZ)
                    old_values = th.randn(BSZ)
                    rewards = th.randn(BSZ)
                    ref = ref_value_loss(new_values,
                                         old_values,
                                         rewards,
                                         clip=clip)
                    loss = ppo.state_value_loss(new_values.view(*shape),
                                                old_values.view(*shape),
                                                rewards.view(*shape),
                                                clip=clip)
                    self.assertAlmostEqual(loss.item(), ref.item(), places=6)


if __name__ == '__main__':
    unittest.main()
