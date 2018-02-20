import argparse
import gym
import os
import cv2
import tensorflow as tf
import numpy as np

from lightsaber.tensorflow.util import initialize
from lightsaber.rl.explorer import LinearDecayExplorer
from lightsaber.rl.replay_buffer import NECReplayBuffer
from actions import get_action_space
from network import make_cnn
from agent import Agent
from trainer import Trainer
from datetime import datetime
from env_wrapper import EnvWrapper
from tensorflow.python.client import timeline

run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
run_metadata = tf.RunMetadata()

def main():
    date = datetime.now().strftime("%Y%m%d%H%M%S")
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='PongDeterministic-v4')
    parser.add_argument('--outdir', type=str, default=date)
    parser.add_argument('--logdir', type=str, default=date)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--load', type=str, default=None)
    parser.add_argument('--final-exploration-frames', type=int, default=10 ** 6)
    parser.add_argument('--final-steps', type=int, default=10 ** 7)
    parser.add_argument('--replay-start-size', type=int, default=5 * 10 ** 4)
    parser.add_argument('--target-update-interval', type=int, default=10 ** 4)
    parser.add_argument('--update-interval', type=int, default=4)
    parser.add_argument('--render', action='store_true')
    args = parser.parse_args()

    # absolute outdir
    outdir = os.path.join(os.path.dirname(__file__), 'results_' + args.outdir)
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    # absolute logdir
    logdir = os.path.join(os.path.dirname(__file__), 'logs/' + args.logdir)

    # v4 4 frames per function call
    def state_preprocess(state):
        state = cv2.cvtColor(state, cv2.COLOR_RGB2GRAY)
        state = cv2.resize(state, (84, 84))
        return state
    env = EnvWrapper(
        gym.make(args.env),
        s_preprocess=state_preprocess,
        r_preprocess=lambda r: np.clip(r, -1, 1)
    )

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
    # GPU SETTINGS
    config = tf.ConfigProto(
        device_count = {'GPU': 1},
        gpu_options=tf.GPUOptions(
            visible_device_list='1',  # gpu device id (from 0)
            # per_process_gpu_memory_fraction=args.gpu_mem,
            # allow_growth=args.gpu_allow_growth
        )
    )

    sess = tf.Session(config=config)
    sess.__enter__()

    agent = Agent(
        model, actions, replay_buffer, explorer, learning_starts=10000,
        run_options=run_options, run_metadata=run_metadata
    )

    initialize()

    saver = tf.train.Saver()
    if args.load is not None:
        saver.restore(sess, args.load)

    trainer = Trainer(
        sess,
        env,
        agent,
        args.update_interval,
        args.render,
        outdir,
        logdir,
        run_metadata
    )

    
    trainer.train(args.final_steps, train=True)

if __name__ == '__main__':
    main()
