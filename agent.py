import build_graph
import numpy as np
import tensorflow as tf
from collections import deque


# state transition cache for N-step update
class Cache:
    def __init__(self, n_step, gamma):
        self.gamma = gamma
        self.states = deque(maxlen=n_step - 1)
        self.actions = deque(maxlen=n_step - 1)
        self.rewards = deque(maxlen=n_step - 1)
        self.encodes = deque(maxlen=n_step - 1)

    def add(self, state, action, reward, encode):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.encodes.append(encode)

    def pop(self, bootstrap_value):
        # calculate N-step value
        R = 0
        for i, r in enumerate(self.rewards):
            R += r * (self.gamma ** i)
        R += bootstrap_value * (self.gamma ** (i + 1))
        # remove oldest values
        reward = self.rewards.popleft()
        state = self.states.popleft()
        action = self.actions.popleft()
        encode = self.encodes.popleft()
        return state, action, encode, R

    def flush(self):
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.encodes.clear()

    def size(self):
        return len(self.states)

class Agent(object):
    def __init__(self,
                network,
                dnds,
                actions,
                state_shape,
                replay_buffer,
                exploration,
                constants,
                phi=lambda s: s,
                run_options=None,
                run_metadata=None):
        self.actions = actions
        self.num_actions = len(actions)

        self.replay_buffer = replay_buffer
        self.exploration = exploration
        self.constants = constants
        self.dnds = dnds
        self.phi = phi
        self.cache = Cache(constants.N_STEP, constants.GAMMA)

        self.last_obs = None
        self.t = 0
        self.t_in_episode = 0

        # TODO: remove
        self.run_options = run_options
        self.run_metadata = run_metadata

        if constants.OPTIMIZER == 'adam':
            optimizer = tf.train.AdamOptimizer(constants.LR)
        else:
            optimizer = tf.train.RMSPropOptimizer(
                learning_rate=constants.LR,
                momentum=constants.MOMENTUM,
                epsilon=constants.EPSILON
            )

        self._act,\
        self._write,\
        self._train = build_graph.build_train(
            encode=network,
            num_actions=self.num_actions,
            state_shape=state_shape,
            optimizer=optimizer,
            dnds=self.dnds,
            key_size=constants.DND_KEY_SIZE,
            grad_clipping=constants.GRAD_CLIPPING,
            run_options=self.run_options,
            run_metadata=self.run_metadata
        )

    # TODO: remove
    def get_epsize(self):
        ''' a helper function to get each episode size of dnds
        '''
        sizes = map(lambda m: min([m.curr_epsize.eval(), m.capacity]), self.dnds)
        return list(sizes)

    # append state transition to DND and replay memory
    def append_experience(self, value):
        state, action, encode, R = self.cache.pop(value)
        state = np.array(state * 255, dtype=np.uint8)
        self.replay_buffer.append(obs_t=state, action=action, value=R)
        self._write[action](encode, R, self.get_epsize())

    def act(self, obs, reward, training=True):
        # preprocess for HWC manner
        obs = self.phi(obs)
        action, values, encoded_state = self._act([obs], self.get_epsize())
        action = action[0]
        encoded_state = encoded_state[0]
        values = values[0]

        # epsilon greedy exploration
        if training:
            action = self.exploration.select_action(
                self.t, action, self.num_actions)
        value = values[action]

        if training and self.t > self.t > self.constants.LEARNING_START_STEP:
            if self.t % self.constants.UPDATE_INTERVAL == 0:
                obs_t, actions, values = self.replay_buffer.sample(
                    self.constants.BATCH_SIZE)
                obs_t = np.array(obs_t / 255., dtype=np.float32)
                td_errors = self._train(obs_t, actions, values, self.get_epsize())

        if training:
            if self.last_obs is not None:
                self.cache.add(
                    self.last_obs,
                    self.last_action,
                    reward,
                    self.last_encoded_state
                )

            if self.t_in_episode >= self.constants.N_STEP:
                self.append_experience(value)

        self.t += 1
        self.t_in_episode += 1
        self.last_obs = obs
        self.last_encoded_state = encoded_state
        self.last_action = action
        return self.actions[action]

    def stop_episode(self, obs, reward, training=True):
        if training:
            self.cache.add(
                self.last_obs,
                self.last_action,
                reward,
                self.last_encoded_state
            )
            while self.cache.size() > 0:
                self.append_experience(0)
        self.last_obs = None
        self.last_action = 0
        self.t_in_episode = 0
        self.cache.flush()
