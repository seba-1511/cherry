import unittest
import random
import torch as th
import cherry as ch
from cherry.td import discount
from cherry.pg import generalized_advantage

GAMMA = 0.5
TAU = 0.9
NUM_SAMPLES = 10
VECTOR_SIZE = 5
TIME_STEPS = 10
NUM_ENVS = 4

"""
TODO: Should test each method to make sure that they properly handle different
      tensor shapes.
TODO: Test temporal_difference.
"""


def close(a, b):
    return (a - b).norm(p=2) <= 1e-6


def discount_rewards(gamma, rewards, dones, bootstrap=0.0):
    """
    Implementation that works with lists.
    """
    R = bootstrap
    discounted = []
    length = len(rewards)
    for t in reversed(range(length)):
        if dones[t]:
            R *= 0.0
        R = rewards[t] + gamma * R
        discounted.insert(0, R)
    return discounted


def generalized_advantage_estimate(gamma,
                                   tau,
                                   rewards,
                                   dones,
                                   values,
                                   next_value):
    msg = 'GAE needs as many rewards, values and dones.'
    assert len(values) == len(rewards) == len(dones), msg
    advantages = []
    advantage = 0
    for i in reversed(range(len(rewards))):
        td_error = rewards[i] + (1.0 - dones[i]) * gamma * next_value - values[i]
        advantage = advantage * tau * gamma * (1.0 - dones[i]) + td_error
        advantages.insert(0, advantage)
        next_value = values[i]
    return advantages


class TestRewards(unittest.TestCase):

    def setUp(self):
        self.replay = ch.ExperienceReplay()

    def tearDown(self):
        pass
    

    def test_vectorized_discount(self):
        state = th.randn(TIME_STEPS, NUM_ENVS, VECTOR_SIZE)
        action = th.randn(TIME_STEPS, NUM_ENVS)
        reward = th.randn(TIME_STEPS, NUM_ENVS)
        boostrap = th.randn(NUM_ENVS)
        done = th.zeros_like(reward)
        for i in list(reversed(range(TIME_STEPS)))[:4]:
            done[i,i%NUM_ENVS] = 1
        

        # Computing the discounted rewards
        # as non-vectorized environment
        nonvec_discounted_rewards = []
        for i in range(NUM_ENVS):
            replay = ch.ExperienceReplay()
            for t in range(TIME_STEPS):
                replay.append(
                    state[t, i, :], action[t, i], 
                    reward[t, i], state[t, i, :], done[t, i]
                )
            nonvec_discounted_rewards.append(
                ch.td.discount(
                    GAMMA, replay.reward(), replay.done(), boostrap[i]
                )
            )
        # Computing the discounted rewards
        # as vectorized environment
        replay = ch.ExperienceReplay(vectorized=True)
        for t in range(TIME_STEPS):
            replay.append(
                state[t, :, :], action[t, :], 
                reward[t, :], state[t, :, :], done[t, :]
            )        
        vec_discounted_rewards = ch.td.discount(
            GAMMA, replay.reward(), replay.done(), boostrap
        )

        for i in range(NUM_ENVS):
            self.assertTrue(
                    close(nonvec_discounted_rewards[i], vec_discounted_rewards[:, i])
            )


    def test_discount(self):
        vector = th.randn(VECTOR_SIZE)
        for i in range(4):
            self.replay.append(vector,
                               vector,
                               8.0,
                               vector,
                               False)
        self.replay.append(vector,
                           vector,
                           8.0,
                           vector,
                           True)
        discounted = discount(GAMMA,
                              self.replay.reward(),
                              self.replay.done(),
                              bootstrap=0)
        ref = th.Tensor([15.5, 15.0, 14.0, 12.0, 8.0]).view(-1, 1)
        self.assertTrue(close(discounted, ref))

        # Test overlapping episodes with bootstrap
        overlap = self.replay[2:] + self.replay[:3]
        overlap_discounted = discount(GAMMA,
                                      overlap.reward(),
                                      overlap.done(),
                                      bootstrap=discounted[3])
        ref = th.cat((discounted[2:], discounted[:3]), dim=0)
        self.assertTrue(close(overlap_discounted, ref))

    def test_generalized_advantage(self):
        vector = th.randn(VECTOR_SIZE)
        for i in range(500):
            self.replay.append(vector,
                               vector,
                               random.random(),
                               vector,
                               False)
        self.replay.done()[-1] += 1
        values = th.randn_like(self.replay.reward())
        rewards = self.replay.reward().view(-1).tolist()
        dones = self.replay.done().view(-1).tolist()
        next_value = random.random()
        ref = generalized_advantage_estimate(GAMMA,
                                             TAU,
                                             rewards,
                                             dones,
                                             values,
                                             next_value)
        advantages = generalized_advantage(GAMMA,
                                           TAU,
                                           self.replay.reward(),
                                           self.replay.done(),
                                           values,
                                           next_value+th.zeros(1))
        ref = th.Tensor(ref).view(advantages.size())
        self.assertTrue(close(ref, advantages))

        # Overlapping episodes
        overlap = self.replay[2:] + self.replay[:3]
        overlap_values = th.cat((values[2:], values[:3]), dim=0)
        overlap_next_value = th.randn(1)
        overlap_adv = generalized_advantage(GAMMA,
                                            TAU,
                                            overlap.reward().double(),
                                            overlap.done().double(),
                                            overlap_values.double(),
                                            overlap_next_value.double())
        values = overlap_values.view(-1).tolist()
        rewards = overlap.reward().view(-1).tolist()
        dones = overlap.done().view(-1).tolist()
        ref = generalized_advantage_estimate(GAMMA,
                                             TAU,
                                             rewards,
                                             dones,
                                             values,
                                             overlap_next_value.item())
        ref = th.Tensor(ref).view(overlap_adv.size()).double()
        self.assertTrue(close(overlap_adv, ref))

    def temporal_difference_test(self):
        pass


if __name__ == '__main__':
    unittest.main()
