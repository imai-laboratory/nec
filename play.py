import argparse
import cv2
import gym
import copy
import os
import numpy as np

from lightsaber.tensorflow.util import initialize
from actions import get_action_space
from actions import get_action_space
from network import make_cnn
from agent import Agent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='PongDeterministic-v0')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--load', type=str, default=None)
    parser.add_argument('--render', action='store_true')
    args = parser.parse_args()

    env = gym.make(args.env)

    actions = get_action_space(args.env)
    n_actions = len(actions)

    model = make_cnn(
        convs=[(32, 8, 4), (64, 4, 2), (64, 3, 1)],
        hiddens=[512]
    )

    sess = tf.Session()
    sess.__enter__()

    agent = Agent(model, n_actions, None, None)

    saver = tf.train.Saver()
    if args.load is not None:
        saver.restore(sess, args.load)

    global_step = 0
    episode = 0

    while True:
        states = np.zeros((args.update_interval, 84, 84), dtype=np.uint8)
        sum_of_rewards = 0
        done = False
        step = 0
        state = env.reset()

        while True:
            if args.render:
                env.render()

            state = cv2.cvtColor(state, cv2.COLOR_RGB2GRAY)
            state = cv2.resize(state, (84, 84))
            states = np.roll(states, 1, axis=0)
            states[0] = state

            action = actions[agent.act(states)]

            if done:
                break

            state, reward, done, info = env.step(action)

            sum_of_rewards += reward
            step += 1
            global_step += 1

        episode += 1

        print('Episode: {}, Step: {}: Reward: {}'.format(
                episode, global_step, sum_of_rewards))

if __name__ == '__main__':
    main()
