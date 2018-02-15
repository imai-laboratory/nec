import argparse
import gym
import os
import tensorflow as tf

from lightsaber.tensorflow.util import initialize
from lightsaber.rl.explorer import LinearDecayExplorer
from lightsaber.rl.replay_buffer import NECReplayBuffer
from actions import get_action_space
from network import make_cnn
from agent import Agent
from trainer import Trainer


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

    # v4 4 frames per function call
    env = gym.make(args.env)

    actions = get_action_space(args.env)

    model = make_cnn(
        convs=[(32, 8, 4), (64, 4, 2), (64, 3, 1)],
        hiddens=[512]
    )

    replay_buffer = NECReplayBuffer(10 ** 5)
    explorer = LinearDecayExplorer(
        final_exploration_step=args.final_exploration_frames
    )

    # Session Configure
    config = tf.ConfigProto(
        device_count={'GPU': 0}
    )
    sess = tf.Session()
    sess.__enter__()

    agent = Agent(
        model, actions, replay_buffer, explorer, learning_starts=10000
    )

    initialize()

    saver = tf.train.Saver()
    if args.load is not None:
        saver.restore(sess, args.load)

    trainer = Trainer(sess, env, agent, args.update_interval, args.render,
                      args.outdir, args.logdir)

    trainer.train(args.final_steps, train=True)


if __name__ == '__main__':
    main()
