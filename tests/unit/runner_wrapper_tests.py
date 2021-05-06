#!/usr/bin/env python3

import logging
import random
import unittest
import numpy as np
import torch as th
import cherry as ch
import cherry.envs as envs

import gym
from gym.vector import AsyncVectorEnv
from gym.envs.unittest import MemorizeDigits

from dummy_env import Dummy


NUM_STEPS = 10


class NullHandler(logging.Handler):

    def emit(self, record):
        pass


class TestRunnerWrapper(unittest.TestCase):

    def setUp(self):
        null_handler = NullHandler()
        self.logger = logging.getLogger('null')
        self.logger.propagate = False
        self.logger.addHandler(null_handler)


    def tearDown(self):
        pass

    def test_vec_episodes(self):
        def test_config(n_envs,
                        n_episodes,
                        base_env,
                        use_torch,
                        use_logger,
                        return_info,
                        retry):
            config = 'n_envs' + str(n_envs) + '-n_eps' + str(n_episodes) \
                    + '-base_env' + str(base_env) \
                    + '-torch' + str(use_torch) + '-logger' + str(use_logger) \
                    + '-info' + str(return_info)
            if isinstance(base_env, str):
                env = vec_env = gym.vector.make(base_env, num_envs=n_envs)
            else:
                def make_env():
                    env = base_env()
                    return env
                env_fns = [make_env for _ in range(n_envs)]
                env = vec_env = AsyncVectorEnv(env_fns)

            if use_logger:
                env = envs.Logger(env, interval=5, logger=self.logger)

            if use_torch:
                env = envs.Torch(env)
                policy = lambda x: ch.totensor(vec_env.action_space.sample())
            else:
                policy = lambda x: vec_env.action_space.sample()

            if return_info:
                agent = lambda x: (policy(x), {'policy': policy(x)[0], 'act': policy(x)})
            else:
                agent = policy

            # Gather experience
            env = envs.Runner(env)
            replay = env.run(agent, episodes=n_episodes)
            if retry:
                replay = env.run(agent, episodes=n_episodes)

            # Pre-compute some shapes
            shape = (len(replay), )
            state_shape = vec_env.observation_space.sample().shape[1:]
            action_shape = np.array(vec_env.action_space.sample())[0].shape
            if len(action_shape) == 0:
                action_shape = (1, )
            done_shape = (1, )

            # Check shapes
            states = replay.state()
            self.assertEqual(states.shape, shape + state_shape, config)
            actions = replay.action()
            self.assertEqual(actions.shape, shape + action_shape, config)
            dones = replay.done()
            self.assertEqual(dones.shape, shape + done_shape, config)
            if return_info:
                policies = replay.policy()
                self.assertEqual(policies.shape, shape + action_shape, config)
                acts = replay.act()
                self.assertEqual(acts.shape, (len(replay), n_envs) + action_shape, config)

        for return_info in [False, True]:
            for use_logger in [False, True]:
                for use_torch in [False, True]:
                    for base_env in [Dummy, MemorizeDigits, 'MemorizeDigits-v0', 'CartPole-v0']:
                        for n_envs in [2, 4]:
                            for n_episodes in [1, 2, 3, 4]:
                                for retry in [False, True]:
                                    test_config(n_envs,
                                                n_episodes,
                                                base_env,
                                                use_torch,
                                                use_logger,
                                                return_info,
                                                retry)

    def test_vec_steps(self):
        """
        - policy return a list of torch vectors
        - policy return a single large tensor
        - policy return (tensor, info)
        - policy return ([tensors], info)
        - policy return (ndarray, info)
        - policy return ([ndarry], info)
        - Using Logger
        - For all cases above: .state(), .done(), .action()
        - Try different envs (not just Dummy)
        - Try 1, 2, 4 envs
        - Check expected shapes of each property
        - Use AsyncVec vs gym.vector.make
        """
        def test_config(n_envs,
                        base_env,
                        use_torch,
                        use_logger,
                        return_info):
            config = 'n_envs' + str(n_envs) + '-base_env' + str(base_env) \
                    + '-torch' + str(use_torch) + '-logger' + str(use_logger) \
                    + '-info' + str(return_info)
            if isinstance(base_env, str):
                env = vec_env = gym.vector.make(base_env, num_envs=n_envs)
            else:
                def make_env():
                    env = base_env()
                    return env
                env_fns = [make_env for _ in range(n_envs)]
                env = vec_env = AsyncVectorEnv(env_fns)

            if use_logger:
                env = envs.Logger(env, interval=5, logger=self.logger)

            if use_torch:
                env = envs.Torch(env)
                policy = lambda x: ch.totensor(vec_env.action_space.sample())
            else:
                policy = lambda x: vec_env.action_space.sample()

            if return_info:
                agent = lambda x: (policy(x), {'policy': policy(x)[0]})
            else:
                agent = policy

            # Gather experience
            env = envs.Runner(env)
            replay = env.run(agent, steps=NUM_STEPS)

            # Pre-compute some shapes
            shape = (NUM_STEPS, n_envs)
            state_shape = vec_env.observation_space.sample()[0]
            if isinstance(state_shape, (int, float)):
                state_shape = tuple()
            else:
                state_shape = state_shape.shape
            action_shape = vec_env.action_space.sample()[0]
            if isinstance(action_shape, (int, float)):
                action_shape = (1, )
            else:
                action_shape = action_shape.shape
            done_shape = (1, )

            # Check shapes
            states = replay.state()
            self.assertEqual(states.shape, shape + state_shape, config)
            actions = replay.action()
            self.assertEqual(actions.shape, shape + action_shape, config)
            dones = replay.done()
            self.assertEqual(dones.shape, shape + done_shape, config)
            if return_info:
                policies = replay.policy()
                self.assertEqual(policies.shape, (NUM_STEPS, ) + action_shape, config)


        for return_info in [False, True]:
            for use_logger in [False, True]:
                for use_torch in [False, True]:
                    for base_env in [Dummy, MemorizeDigits, 'MemorizeDigits-v0', 'CartPole-v0']:
                        for n_envs in [2, 4]:
                            test_config(n_envs,
                                        base_env,
                                        use_torch,
                                        use_logger,
                                        return_info)
                                


if __name__ == '__main__':
    unittest.main()
