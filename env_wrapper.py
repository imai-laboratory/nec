import numpy as np


class EnvWrapper:
    def __init__(self, env, r_preprocess=lambda s: s, s_preprocess=lambda r: r):
        self.env = env
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self.r_preprocessor = r_preprocessor
        self.s_preprocessor = s_preprocessor

    def step(self, action):
        state, reward, done, info = self.env.step(action)
        state = self.s_preprocess(state)
        reward = self.r_preprocess(reward)
        return state, reward, done, info

    def reset(self):
        state = self.env.reset()
        return self.s_preprocess(state)

    def render(self):
        self.env.render()

