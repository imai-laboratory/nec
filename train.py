import argparse
import cv2
import gym
import copy
import os
import numpy as np
import tensorflow as tf

from lightsaber.tensorflow.util import initialize
from lightsaber.rl.explorer import LinearDecayExplorer
from lightsaber.rl.replay_buffer import NECReplayBuffer
from actions import get_action_space
from network import make_cnn
from agent import Agent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='PongDeterministic-v4')
    parser.add_argument('--outdir', type=str, default=None)
    parser.add_argument('--logdir', type=str, default=None)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--load', type=str, default=None)
    parser.add_argument('--final-exploration-frames',
                        type=int, default=10 ** 6)
    parser.add_argument('--final-steps', type=int, default=10 ** 7)
    parser.add_argument('--replay-start-size', type=int, default=5 * 10 ** 4)
    parser.add_argument('--target-update-interval',
                        type=int, default=10 ** 4)
    parser.add_argument('--update-interval', type=int, default=4)
    parser.add_argument('--render', action='store_true')
    args = parser.parse_args()

    if args.outdir is None:
        args.outdir = os.path.join(os.path.dirname(__file__), 'results')
        if not os.path.exists(args.outdir):
            os.makedirs(args.outdir)
    if args.logdir is None:
        args.logdir = os.path.join(os.path.dirname(__file__), 'logs')

    env = gym.make(args.env)

    actions = get_action_space(args.env)
    n_actions = len(actions)

    model = make_cnn(
        convs=[(32, 8, 4), (64, 4, 2), (64, 3, 1)],
        hiddens=[512]
    )
    replay_buffer = NECReplayBuffer(10 ** 5)
    explorer = LinearDecayExplorer(final_exploration_step=args.final_exploration_frames)

    sess = tf.Session()
    sess.__enter__()

    agent = Agent(model, n_actions, replay_buffer, explorer, learning_starts=10000)

    initialize()

    saver = tf.train.Saver()
    if args.load is not None:
        saver.restore(sess, args.load)

    reward_summary = tf.placeholder(tf.int32, (), name='reward_summary')
    tf.summary.scalar('reward_summary', reward_summary)
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(args.logdir, sess.graph)

    global_step = 0
    episode = 0

    while True:
        states = np.zeros((args.update_interval, 84, 84), dtype=np.uint8)
        reward = 0
        done = False
        clipped_reward = 0
        sum_of_rewards = 0
        step = 0
        state = env.reset()

        while True:
            if args.render:
                env.render()

            state = cv2.cvtColor(state, cv2.COLOR_RGB2GRAY)
            state = cv2.resize(state, (84, 84))
            states = np.roll(states, 1, axis=0)
            states[0] = state

            if done:
                summary, _ = sess.run([merged, reward_summary], feed_dict={reward_summary: sum_of_rewards})
                train_writer.add_summary(summary, global_step)
                agent.stop_episode_and_train(np.transpose(states, [1, 2, 0]), clipped_reward)
                break

            action = actions[agent.act_and_train(np.transpose(states, [1, 2, 0]), clipped_reward)]

            state, reward, done, info = env.step(action)

            if reward > 0:
                clipped_reward = 1.0
            elif reward < 0:
                clipped_reward = -1.0
            else:
                clipped_reward = 0.0
            sum_of_rewards += reward
            step += 1
            global_step += 1

            if global_step % 10 ** 6 == 0:
                path = os.path.join(args.outdir, '{}/model.ckpt'.format(global_step))
                saver.save(sess, path)

        episode += 1

        print('Episode: {}, Step: {}: Reward: {}'.format(
                episode, global_step, sum_of_rewards))

        if args.final_steps < global_step:
            break

if __name__ == '__main__':
    main()
