import build_graph
import numpy as np
import tensorflow as tf
from dnd import DND
from collections import deque


class Agent(object):
    def __init__(
            self, encode, actions,
            replay_buffer, exploration,
            options, run_options=None, run_metadata=None
    ):
        self.actions = actions
        self.num_actions = len(actions)

        self.options = options
        self.last_obs = None
        self.t = 0
        self.t_in_episode = 0
        self.exploration = exploration
        self.replay_buffer = replay_buffer
        self.reward_cache = deque(maxlen=options.n_step - 1)
        self.state_cache = deque(maxlen=options.n_step - 1)
        self.action_cache = deque(maxlen=options.n_step - 1)
        self.encoded_state_cache = deque(maxlen=options.n_step - 1)
        self.dnds = []

        # TODO: remove
        self.run_options = run_options
        self.run_metadata = run_metadata

        # TODO: list comprehension
        for i in range(self.num_actions):
            dnd = DND(options.encoded_size, options.capacity, options.p)
            dnd._init_vars()
            self.dnds.append(dnd)

        act, write, train = build_graph.build_train(
            encode=encode,
            num_actions=self.num_actions,
            optimizer=tf.train.RMSPropOptimizer(
                learning_rate=options.lr,
                momentum=options.momentum,
                epsilon=options.epsilon
            ),
            dnds=self.dnds,
            options=options,
            run_options=self.run_options,
            run_metadata=self.run_metadata
        )
        self._act = act
        self._write = write
        self._train = train

    # TODO: remove
    def get_epsize(self):
        ''' a helper function to get each episode size of dnds
        '''
        rvals = [
            min([dnd.curr_epsize.eval(), dnd.capacity]) for dnd in self.dnds
        ]
        return rvals

    def append_experience(self, value):
        ''' add experiences to DND
        '''
        # R: Return
        R = 0
        for i, r in enumerate(self.reward_cache):
            R += r * (self.options.gamma ** i)
        R += value * (self.options.gamma ** (i + 1))

        obs_t = self.state_cache[0]
        encoded_state = self.encoded_state_cache[0]
        action = self.action_cache[0]
        self.replay_buffer.append(obs_t=obs_t, action=action, value=R)
        self._write[action](encoded_state, R, self.get_epsize())

    def act(self, obs):
        normalized_obs = np.zeros([1] + list(self.options.in_shape), dtype=np.float32)
        normalized_obs[0] = np.array(obs, dtype=np.float32) / 255.0
        action = self._act(normalized_obs, self.get_epsize())[0]
        return action

    def act_and_train(self, obs, reward):
        normalized_obs = np.zeros([1] + list(self.options.in_shape), dtype=np.float32)
        normalized_obs[0] = np.array(obs, dtype=np.float32) / 255.0
        action, values, encoded_state = self._act(normalized_obs, self.get_epsize())
        action = action[0]
        encoded_state = encoded_state[0]
        values = values[0]
        action = self.exploration.select_action(self.t, action, self.num_actions)
        value = values[action]

        if self.t > self.options.learning_starts and self.t % self.options.train_freq == 0:
            obs_t, actions, values = self.replay_buffer.sample(self.options.batch_size)
            obs_t = np.array(obs_t, dtype=np.float32) / 255.0
            td_errors = self._train(obs_t, actions, values, self.get_epsize())

        if self.last_obs is not None:
            self.reward_cache.append(reward)
            self.state_cache.append(self.last_obs)
            self.action_cache.append(self.last_action)
            self.encoded_state_cache.append(self.last_encoded_state)

        if self.t_in_episode >= self.options.n_step:
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
