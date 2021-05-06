#!/usr/bin/env python3

import unittest
import random
import numpy as np
import torch as th
import cherry as ch
import os
import copy


NUM_SAMPLES = 100
VECTOR_SIZE = 5


def close(a, b):
    return (a-b).norm(p=2) <= 1e-8


class TestExperienceReplay(unittest.TestCase):

    def setUp(self):
        self.replay = ch.ExperienceReplay()

    def tearDown(self):
        pass

    def test_empty(self):
        vector = np.random.rand(VECTOR_SIZE)
        for i in range(NUM_SAMPLES):
            self.replay.append(vector,
                               vector,
                               i,
                               vector,
                               False,
                               vector=vector)
        self.replay.empty()
        self.assertEqual(len(self.replay._storage), 0)

    def test_len(self):
        vector = np.random.rand(VECTOR_SIZE)
        for i in range(NUM_SAMPLES):
            self.replay.append(vector,
                               vector,
                               i,
                               vector,
                               False,
                               vector=vector)
        self.assertEqual(len(self.replay), NUM_SAMPLES)
        self.assertEqual(len(self.replay._storage), NUM_SAMPLES)

    def test_add_numpy(self):
        for shape in [(VECTOR_SIZE,), (1, VECTOR_SIZE)]:
            vector = np.random.rand(*shape)
            for i in range(NUM_SAMPLES):
                self.replay.append(vector,
                                   vector,
                                   i,
                                   vector,
                                   False,
                                   vector=vector)
            ref_size = th.Size([NUM_SAMPLES, VECTOR_SIZE])
            self.assertTrue(isinstance(self.replay.state(), th.Tensor))
            self.assertEqual(self.replay.state().size(), ref_size)
            self.assertTrue(isinstance(self.replay.action(), th.Tensor))
            self.assertEqual(self.replay.action().size(), ref_size)
            self.assertTrue(isinstance(self.replay.reward(), th.Tensor))
            self.assertTrue(isinstance(self.replay.next_state(), th.Tensor))
            self.assertEqual(self.replay.next_state().size(), ref_size)
            self.assertTrue(isinstance(self.replay.done(), th.Tensor))
            self.replay.empty()

    def test_add_torch(self):
        for shape in [(VECTOR_SIZE, ), (1, VECTOR_SIZE)]:
            vector = th.randn(*shape)
            for i in range(NUM_SAMPLES):
                self.replay.append(vector,
                                   vector,
                                   i,
                                   vector,
                                   False,
                                   vector=vector)
            ref_size = th.Size([NUM_SAMPLES, VECTOR_SIZE])
            self.assertTrue(isinstance(self.replay.state(), th.Tensor))
            self.assertEqual(self.replay.state().size(), ref_size)
            self.assertTrue(isinstance(self.replay.action(), th.Tensor))
            self.assertEqual(self.replay.action().size(), ref_size)
            self.assertTrue(isinstance(self.replay.reward(), th.Tensor))
            self.assertTrue(isinstance(self.replay.next_state(), th.Tensor))
            self.assertEqual(self.replay.next_state().size(), ref_size)
            self.assertTrue(isinstance(self.replay.done(), th.Tensor))
            self.replay.empty()

    def test_slice(self):
        # Fill replay
        count = 0
        for shape in [(VECTOR_SIZE, ), (1, VECTOR_SIZE)]:
            vector = th.randn(*shape)
            for i in range(NUM_SAMPLES):
                count += 1
                self.replay.append(vector,
                                   vector,
                                   i,
                                   vector,
                                   random.choice([False, True]),
                                   vector=vector,
                                   id=count)

        subsample = self.replay[0:len(self.replay)//2]
        self.assertTrue(isinstance(subsample, ch.ExperienceReplay))
        self.assertEqual(len(subsample), len(self.replay)//2)
        self.assertTrue(isinstance(self.replay[0], ch.Transition))
        subsample = self.replay[-1:]
        self.assertEqual(len(subsample), 1)

    def test_sample(self):
        # Test empty
        sample = self.replay.sample()
        self.assertEqual(len(sample), 0)
        self.assertTrue(isinstance(self.replay, ch.ExperienceReplay))

        # Fill replay
        count = 0
        for shape in [(VECTOR_SIZE, ), (1, VECTOR_SIZE)]:
            vector = th.randn(*shape)
            for i in range(NUM_SAMPLES):
                count += 1
                self.replay.append(vector,
                                   vector,
                                   i,
                                   vector,
                                   random.choice([False, True]),
                                   vector=vector,
                                   id=count)
            for _ in range(30):
                # Test default arguments
                sample = self.replay.sample()
                self.assertEqual(len(sample), 1)
                # Test size
                sample = self.replay.sample(size=NUM_SAMPLES//2)
                self.assertEqual(len(sample), NUM_SAMPLES//2)
                # Test contiguous
                sample = self.replay.sample(size=NUM_SAMPLES//3,
                                            contiguous=True)
                ids = sample.id()
                for i, id in enumerate(ids[:-1]):
                    self.assertEqual(id + 1, ids[i+1])

                # Test single episode
                sample = self.replay.sample(size=1, episodes=True)
                self.assertTrue(bool(sample[-1].done.item()))
                for i, sars in enumerate(sample[:-1]):
                    self.assertTrue(not sample[i].done)
                    self.assertEqual(sars.id + 1,
                                     sample[i+1].id)

                # Test multiple episodes
                total_episodes = self.replay.done().sum().int().item()
                for num_episodes in [total_episodes, total_episodes//2, 1]:
                    sample = self.replay.sample(size=num_episodes,
                                                episodes=True)
                    num_sampled_episodes = sample.done().sum().int().item()
                    self.assertEqual(num_sampled_episodes, num_episodes)

                # Test multiple contiguous episodes
                total_episodes = self.replay.done().sum().int().item()
                for num_episodes in [total_episodes, total_episodes//2, 1]:
                    sample = self.replay.sample(size=num_episodes,
                                                episodes=True)
                    num_sampled_episodes = sample.done().sum().int().item()
                    self.assertEqual(num_sampled_episodes, num_episodes)
                    for i, sars in enumerate(sample[:-1]):
                        if not sars.done:
                            self.assertEqual(sample[i].id+1,
                                             sample[i+1].id)

    def test_append(self):
        new_replay = ch.ExperienceReplay()
        vector = np.random.rand(VECTOR_SIZE)
        for i in range(NUM_SAMPLES):
            self.replay.append(vector,
                               vector,
                               i,
                               vector,
                               False,
                               vector=vector)
            new_replay.append(vector,
                              vector,
                              i,
                              vector,
                              False,
                              vector=vector)
        self.assertEqual(len(self.replay), len(new_replay))
        new_replay = self.replay + new_replay
        self.assertEqual(NUM_SAMPLES * 2, len(new_replay))
        self.replay += new_replay
        self.assertEqual(NUM_SAMPLES * 3, len(self.replay))

    def test_save_and_load(self):
        old_replay = self.replay
        vector = np.random.rand(VECTOR_SIZE)
        for i in range(NUM_SAMPLES):
            old_replay.append(vector,
                              vector,
                              i,
                              vector,
                              False,
                              vector=vector)
        # save the old file
        old_replay.save('testing_temp_file.pt')

        # load the saved file to a new file
        new_replay = ch.ExperienceReplay()
        new_replay.load('testing_temp_file.pt')

        # check size
        self.assertEqual(len(old_replay._storage),
                         len(new_replay._storage))
        self.assertEqual(len(old_replay.state()),
                         len(new_replay.state()))
        self.assertEqual(len(old_replay.action()),
                         len(new_replay.action()))
        self.assertEqual(len(old_replay.reward()),
                         len(new_replay.reward()))
        self.assertEqual(len(old_replay.next_state()),
                         len(new_replay.next_state()))
        self.assertEqual(len(old_replay.done()),
                         len(new_replay.done()))
        self.assertEqual(len(old_replay.vector()),
                         len(new_replay.vector()))

        # check content
        for a, b in zip(old_replay, new_replay):
            self.assertTrue(close(a.state, b.state))
            self.assertTrue(close(a.action, b.action))
            self.assertTrue(close(a.reward, b.reward))
            self.assertTrue(close(a.next_state, b.next_state))
            self.assertTrue(close(a.done, b.done))
            self.assertTrue(close(a.vector, b.vector))

        os.remove('testing_temp_file.pt')

    def test_replay_myattr(self):
        standard_replay = self.replay
        vector = np.random.rand(VECTOR_SIZE)

        # a random tensor to be stuffed in
        test_tensor = th.randn(3, 3, dtype=th.double)

        # initialization, stuff just tensors in
        # and the results type should still be tensor
        for i in range(NUM_SAMPLES):
            standard_replay.append(vector,
                                   vector,
                                   i,
                                   vector,
                                   False,
                                   test=test_tensor)
        self.assertTrue(isinstance(standard_replay.test(), th.Tensor))

    def test_slices(self):
        for i in range(NUM_SAMPLES):
            self.replay.append(th.randn(VECTOR_SIZE),
                               th.randn(VECTOR_SIZE),
                               i,
                               th.randn(VECTOR_SIZE),
                               False,
                               vector=th.randn(VECTOR_SIZE))

        sliced = self.replay[0:-3]
        self.assertEqual(len(sliced), len(self.replay) - 3)
        for sars, sars_ in zip(self.replay, sliced):
            self.assertTrue(close(sars.state, sars_.state))
            self.assertTrue(close(sars.action, sars_.action))
            self.assertTrue(close(sars.reward, sars_.reward))
            self.assertTrue(close(sars.next_state, sars_.next_state))
            self.assertTrue(close(sars.vector, sars_.vector))

    def test_to_device(self):
        for i in range(NUM_SAMPLES):
            self.replay.append(th.randn(VECTOR_SIZE),
                               th.randn(VECTOR_SIZE),
                               i,
                               th.randn(VECTOR_SIZE),
                               False,
                               vector=th.randn(VECTOR_SIZE))

        # Test function calls
        replay = self.replay.to(None)
        self.assertEqual(len(replay), len(self.replay))
        replay = self.replay.to('cpu')
        self.assertEqual(len(replay), len(self.replay))
        replay = self.replay.cpu()
        self.assertEqual(len(replay), len(self.replay))

        for cr, sr in zip(replay, self.replay):
            self.assertTrue(close(cr.state, sr.state))
            self.assertTrue(close(cr.action, sr.action))
            self.assertTrue(close(cr.next_state, sr.next_state))
            self.assertTrue(close(cr.reward, sr.reward))
            self.assertTrue(close(cr.vector, sr.vector))
            self.assertTrue(close(cr.vector, sr.vector))

        # Test cuda
        if th.cuda.is_available():
            cuda_replay = self.replay.cuda()

            self.assertEqual(len(cuda_replay), len(self.replay))

            for cr, sr in zip(cuda_replay, self.replay):
                self.assertTrue(close(cr.state, sr.state.cuda()))
                self.assertTrue(close(cr.action, sr.action.cuda()))
                self.assertTrue(close(cr.next_state, sr.next_state.cuda()))
                self.assertTrue(close(cr.reward, sr.reward.cuda()))
                self.assertTrue(close(cr.vector, sr.vector.cuda()))
                self.assertTrue(close(cr.vector, sr.vector.cuda()))

            replay = cuda_replay.to('cpu')

            self.assertEqual(len(replay), len(self.replay))

            for cr, sr in zip(replay, self.replay):
                self.assertTrue(close(cr.state, sr.state))
                self.assertTrue(close(cr.action, sr.action))
                self.assertTrue(close(cr.next_state, sr.next_state))
                self.assertTrue(close(cr.reward, sr.reward))
                self.assertTrue(close(cr.vector, sr.vector))
                self.assertTrue(close(cr.vector, sr.vector))

    def test_to_dtype(self):
        for i in range(NUM_SAMPLES):
            self.replay.append(th.randn(VECTOR_SIZE),
                               th.randn(VECTOR_SIZE),
                               i,
                               th.randn(VECTOR_SIZE),
                               False,
                               vector=th.randn(VECTOR_SIZE))
        f32 = self.replay.to(th.float32)
        f64 = self.replay.to(th.float64)
        i32 = self.replay.to(th.int32)
        i64 = self.replay.to(th.int64)

    def test_half_double(self):
        for i in range(NUM_SAMPLES):
            self.replay.append(th.randn(VECTOR_SIZE),
                               th.randn(VECTOR_SIZE),
                               i,
                               th.randn(VECTOR_SIZE),
                               False,
                               vector=th.randn(VECTOR_SIZE))
        half = self.replay.half()
        half_dtype = self.replay[0].state.half().dtype
        for sars in half:
            self.assertTrue(sars.state.dtype == half_dtype)

        double = self.replay.double()
        double_dtype = self.replay[0].state.double().dtype
        for sars in double:
            self.assertTrue(sars.state.dtype == double_dtype)

        if th.cuda.is_available():
            cuda_replay = self.replay.cuda()
            half = cuda_replay.half()
            half_dtype = cuda_replay[0].state.half().dtype
            for sars in half:
                self.assertTrue(sars.state.dtype == half_dtype)
            double = cuda_replay.double()
            double_dtype = cuda_replay[0].state.double().dtype
            for sars in double:
                self.assertTrue(sars.state.dtype == double_dtype)

    def test_flatten(self):
        def original_flatten(replay):  # slow but correct
            if not replay.vectorized:
                return replay
            flat_replay = ch.ExperienceReplay(device=replay.device, vectorized=False)
            for sars in replay._storage:
                for i in range(sars.done.shape[0]):
                    transition = {
                        field: getattr(sars, field)[i] for field in sars._fields
                    }
                    # need to add dimension back because of indexing above.
                    transition = {
                        k: v.unsqueeze(0)
                        if ch._utils._istensorable(v) else v
                        for k, v in transition.items()
                    }
                    flat_replay.append(**transition)
            return flat_replay

        num_envs = 8
        batch_size = 2^5
        replay_size = 2^6
        s_shape = (num_envs, 9, 84, 84)
        a_shape = (num_envs, 84)

        for device in ['cpu', 'cuda']:
            if not th.cuda.is_available() and device == 'cuda':
                continue

            # generate data
            replay = ch.ExperienceReplay(vectorized=True)
            for step in range(replay_size):
                action = th.randn(*a_shape)
                state = th.randn(*s_shape)
                done = th.randint(low=0, high=1, size=(num_envs, 1))
                reward = th.randn((num_envs, 1))
                info = {
                    'success': [0.0, ] * num_envs,
                    'numpy': np.random.randn(num_envs, 23, 4)
                        }
                replay.append(state, action, reward, state, done, **info)
            replay.to(device)

            # test the two flatten are identical
            for batch in [replay, replay.sample(batch_size)]:
                b1 = original_flatten(batch)
                b2 = batch.flatten()
                for sars1, sars2 in zip(b1, b2):
                    for field in sars1._fields:
                        val1 = getattr(sars1, field)
                        val2 = getattr(sars2, field)
                        self.assertTrue(
                            (val1.double() - val2.double()).norm().item() < 1e-8,
                            'flatten values mismatch',
                        )
                        self.assertTrue(
                            val1.shape == val2.shape,
                            'flatten shape mismatch',
                        )
                        self.assertTrue(
                            val1.device == val2.device,
                            'flatten device misatch',
                        )


if __name__ == '__main__':
    unittest.main()
