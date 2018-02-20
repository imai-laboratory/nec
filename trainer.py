import numpy as np
import tensorflow as tf
import os
from tensorflow.python.client import timeline

class Trainer:
    def __init__(
            self, sess, env, agent,
            update_interval, render,
            outdir, log_dir,
            run_metadata=None
    ):
        ''' a Trainer class initalizer
        ARGS:
            sess (object): session
            env (object): environment
            agent (object): agent
            update_interval (int): update every number of frames
        '''
        self.sess = sess
        self.env = env
        self.agent = agent

        # Training Options
        self.update_interval = update_interval
        self.render = render
        self.outdir = outdir
        self.log_dir = log_dir

        self.save_model_per = 10 ** 6

        # Summarizer
        self.reward_summary = tf.placeholder(
            tf.int32, (), name='reward_summary')
        tf.summary.scalar('reward_summary', self.reward_summary)
        self.merged = tf.summary.merge_all()
        self.train_writer = tf.summary.FileWriter(self.log_dir, sess.graph)

        # Trainer global
        self.global_step = 0
        self.episode = 0

        # TODO: remove
        self.run_metadata = run_metadata

    def train(self, final_steps, train=True):
        while True:
            # states are in RGBD?
            states = np.zeros((self.update_interval, 84, 84), dtype=np.uint8)
            reward = 0
            done = False
            sum_of_rewards = 0
            step = 0
            state = self.env.reset()

            while True:
                if self.render:
                    self.env.render()

                states = np.roll(states, 1, axis=0)
                states[0] = state

                if done:
                    summary, _ = self.sess.run(
                        [self.merged, self.reward_summary],
                        feed_dict={self.reward_summary: sum_of_rewards}
                    )
                    self.train_writer.add_summary(summary, self.global_step)
                    self.agent.stop_episode_and_train(
                        np.transpose(states, [1, 2, 0]), reward
                    )
                    break

                action = self.agent.actions[
                    self.agent.act_and_train(
                        np.transpose(states, [1, 2, 0]), reward
                    )
                ]

                state, reward, done, info = self.env.step(action)
                sum_of_rewards += reward
                step += 1

                if train:
                    self.global_step += 1
                    if self.global_step % self.save_model_per == 0:
                        self.save_model()
            if train:
                self.episode += 1
            print('Episode: {}, Step: {}: Reward: {}'.format(
                    self.episode, self.global_step, sum_of_rewards))

            # TODO: remove
            step_stats = self.run_metadata.step_stats
            tl = timeline.Timeline(step_stats)
            ctf = tl.generate_chrome_trace_format(show_memory=False,
                                              show_dataflow=True)
            with open("timeline.json", "w") as f:
                print('write')
                f.write(ctf)

            if final_steps < self.global_step:
                break

    def save_model(self):
        ''' saves a model
        '''
        path = os.path.join(
            self.outdir, '{}/model.ckpt'.format(self.global_step)
        )
        self.saver.save(self.sess, path)
