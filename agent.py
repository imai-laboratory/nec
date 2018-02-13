import network
import build_graph
import lightsaber.tensorflow.util as util
import numpy as np
import tensorflow as tf
from dnd  import DND
from collections import deque


class Agent(object):
    def __init__(
            self, encode, actions,
            replay_buffer, exploration, lr=2.5e-4, batch_size=32,
            train_freq=16, learning_starts=10000, gamma=0.99, n_step=100
    ):
        self.batch_size = batch_size
        self.train_freq = train_freq
        self.actions = actions
        self.num_actions = len(actions)
        self.learning_starts = learning_starts
        self.n_step = n_step
        self.gamma = gamma
        self.last_obs = None
        self.t = 0
        self.t_in_episode = 0
        self.exploration = exploration
        self.replay_buffer = replay_buffer
        self.reward_cache = deque(maxlen=n_step - 1)
        self.state_cache = deque(maxlen=n_step - 1)
        self.action_cache = deque(maxlen=n_step - 1)
        self.encoded_state_cache = deque(maxlen=n_step - 1)
        self.dnds = []

        for i in range(self.num_actions):
            dnd = DND()
            dnd._init_vars()
            self.dnds.append(dnd)

        act, write, train, q_values = build_graph.build_train(
            encode=encode,
            num_actions=self.num_actions,
            optimizer=tf.train.RMSPropOptimizer(
                learning_rate=lr, momentum=0.95, epsilon=1e-2
            ),
            dnds=self.dnds,
            gamma=gamma,
            grad_norm_clipping=10.0
        )
        self._act = act
        self._write = write
        self._train = train
        self._q_values = q_values

    def get_epsize(self):
        rvals = [
            min([dnd.curr_epsize.eval, 10 ** 5]) for dnd in self.dnds
        ]
        return rvals

    def append_experience(self, value):
        ''' add experiences to DND
        '''
        R = 0
        for i, r in enumerate(self.reward_cache):
            R += r * (self.gamma ** i)
        R += value * (self.gamma ** (i + 1))

        obs_t = self.state_cache[0]
        encoded_state = self.encoded_state_cache[0]
        action = self.action_cache[0]
        self.replay_buffer.append(obs_t=obs_t, action=action, value=R)
        self._write[action](encoded_state, value, self.get_epsize())

    def act(self, obs):
        normalized_obs = np.zeros((1, 84, 84, 4), dtype=np.float32)
        normalized_obs[0] = np.array(obs, dtype=np.float32) / 255.0
        action = self._act(normalized_obs)[0]
        return action

    def act_and_train(self, obs, reward):
        normalized_obs = np.zeros((1, 84, 84, 4), dtype=np.float32)
        normalized_obs[0] = np.array(obs, dtype=np.float32) / 255.0
        action, encoded_state = self._act(normalized_obs)
        action = action[0]
        encoded_state = encoded_state[0]
        action = self.exploration.select_action(self.t, action, self.num_actions)
        value = self._q_values(normalized_obs)[0][action]

        if self.t > self.learning_starts and self.t % self.train_freq == 0:
            obs_t, actions, values = self.replay_buffer.sample(self.batch_size)
            obs_t = np.array(obs_t, dtype=np.float32) / 255.0
            td_errors = self._train(obs_t, actions, values)

        if self.last_obs is not None:
            self.reward_cache.append(reward)
            self.state_cache.append(self.last_obs)
            self.action_cache.append(self.last_action)
            self.encoded_state_cache.append(self.last_encoded_state)

        if self.t_in_episode >= self.n_step:
            self.append_experience(value)

        self.t += 1
        self.t_in_episode += 1
        self.last_obs = obs
        self.last_encoded_state = encoded_state
        self.last_action = action
        return action

    def stop_episode_and_train(self, obs, reward):
        self.reward_cache.append(reward)
        self.state_cache.append(self.last_obs)
        self.action_cache.append(self.last_action)
        self.encoded_state_cache.append(self.last_encoded_state)
        while len(self.reward_cache) > 0:
            self.append_experience(0)
            self.reward_cache.popleft()
            self.state_cache.popleft()
            self.action_cache.popleft()
            self.encoded_state_cache.popleft()
        self.stop_episode()

    def stop_episode(self):
        self.last_obs = None
        self.last_action = 0
        self.t_in_episode = 0
        self.reward_cache.clear()
        self.state_cache.clear()
        self.action_cache.clear()
        self.encoded_state_cache.clear()
