#!/usr/bin/env python3

from statistics import mean, pstdev

from .base import Wrapper

import cherry as ch
import numpy as np


class Logger(Wrapper):

    """
    Tracks and prints some common statistics about the environment.
    """

    def __init__(self, env, interval=1000, episode_interval=10, title=None, logger=None):
        super(Logger, self).__init__(env)
        self.num_steps = 0
        self.num_episodes = 0
        self.all_rewards = []
        self.all_dones = []
        self.interval = interval
        self.ep_interval = episode_interval
        self.values = {}
        self.values_idx = {}

        if title is None:
            if hasattr(env, 'spec') and hasattr(env.spec, 'id'):
                title = env.spec.id
            else:
                title = ''
        self.title = title

        if logger is None:
            logger = ch.debug.logger
        self.logger = logger

    def _episodes_length_rewards(self, rewards, dones):
        """
        When dealing with array rewards and dones (as for VecEnv) the length
        and rewards are only computed on the first dimension.
        (i.e. the first sub-process.)
        """
        episode_rewards = []
        episode_lengths = []
        accum = 0.0
        length = 0
        for r, d in zip(rewards, dones):
            if not isinstance(d, bool):
                d = bool(d.flat[0])
                r = float(r.flat[0])
            if not d:
                accum += r
                length += 1
            else:
                episode_rewards.append(accum)
                episode_lengths.append(length)
                accum = 0.0
                length = 0
        if length > 0:
            episode_rewards.append(accum)
            episode_lengths.append(length)
        return episode_rewards, episode_lengths

    def _episodes_stats(self):
        # Find the last episodes
        start = end = count = 0
        for i, d in reversed(list(enumerate(self.all_dones))):
            if not isinstance(d, bool):
                d = d.flat[0]
            if d:
                count += 1
                if end == 0:
                    end = i
                if count >= self.ep_interval + 1:
                    start = i + 1
        # Compute stats
        rewards = self.all_rewards[start:end]
        dones = self.all_dones[start:end]
        rewards, lengths = self._episodes_length_rewards(rewards, dones)
        stats = {
            'episode_rewards': rewards if len(rewards) else [0.0],
            'episode_lengths': lengths if len(lengths) else [0.0],
        }
        return stats

    def _steps_stats(self, update_index=False):
        rewards = self.all_rewards[-self.interval:]
        dones = self.all_dones[-self.interval:]
        rewards, lengths = self._episodes_length_rewards(rewards, dones)
        stats = {
            'episode_rewards': rewards,
            'episode_lengths': lengths,
            'num_episodes': len(rewards),
        }
        for key in self.values.keys():
            idx = self.values_idx[key]
            stats[key] = self.values[key][idx:]
            if update_index:
                self.values_idx[key] = len(self.values[key]) - 1
        return stats

    def stats(self):
        # Compute statistics
        ep_stats = self._episodes_stats()
        steps_stats = self._steps_stats(update_index=True)

        # Overall stats
        num_logs = len(self.all_rewards) // self.interval
        msg = '-' * 20 + ' ' + self.title + ' Log ' + str(num_logs) + ' ' + '-' * 20 + '\n'
        msg += 'Overall:' + '\n'
        msg += '- Steps: ' + str(self.num_steps) + '\n'
        msg += '- Episodes: ' + str(self.num_episodes) + '\n'

        # Episodes stats
        msg += 'Last ' + str(self.ep_interval) + ' Episodes:' + '\n'
        msg += '- Mean episode length: ' + '%.2f' % mean(ep_stats['episode_lengths'])
        msg += ' +/- ' + '%.2f' % pstdev(ep_stats['episode_lengths']) + '\n'
        msg += '- Mean episode reward: ' + '%.2f' % mean(ep_stats['episode_rewards'])
        msg += ' +/- ' + '%.2f' % pstdev(ep_stats['episode_rewards']) + '\n'

        # Steps stats
        msg += 'Last ' + str(self.interval) + ' Steps:' + '\n'
        msg += '- Episodes: ' + str(steps_stats['num_episodes']) + '\n'
        msg += '- Mean episode length: ' + '%.2f' % mean(steps_stats['episode_lengths'])
        msg += ' +/- ' + '%.2f' % pstdev(steps_stats['episode_lengths']) + '\n'
        msg += '- Mean episode reward: ' + '%.2f' % mean(steps_stats['episode_rewards'])
        msg += ' +/- ' + '%.2f' % pstdev(steps_stats['episode_rewards']) + '\n'
        for key in self.values.keys():
            msg += '- Mean ' + key + ': ' + '%.2f' % mean(steps_stats[key])
            msg += ' +/- ' + '%.2f' % pstdev(steps_stats[key]) + '\n'
        return msg, ep_stats, steps_stats

    def reset(self, *args, **kwargs):
        return self.env.reset(*args, **kwargs)

    def log(self, key, value):
        if key not in self.values:
            self.values[key] = []
            self.values_idx[key] = 0
            setattr(self, key, self.values[key])
        self.values[key].append(value)

    def step(self, *args, **kwargs):
        state, reward, done, info = self.env.step(*args, **kwargs)
        self.all_rewards.append(reward)
        self.all_dones.append(done)
        self.num_steps += 1
        if self.interval > 0 and self.num_steps % self.interval == 0:
            msg, ep_stats, steps_stats = self.stats()
            if self.is_vectorized:
                info[0]['logger_steps_stats'] = steps_stats
                info[0]['logger_ep_stats'] = ep_stats
            else:
                info['logger_steps_stats'] = steps_stats
                info['logger_ep_stats'] = ep_stats
            self.logger.info(msg)
        if isinstance(done, (bool, np.bool_)):
            if done:
                self.num_episodes += 1
        else:
            self.num_episodes += sum(done)
        return state, reward, done, info
