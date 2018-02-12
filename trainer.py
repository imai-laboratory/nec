import numpy as np
import tensorflow as tf
import cv2
import os


class Trainer:
    def __init__(
            self, sess, env, agent,
            update_interval, render,
            outdir, log_dir,
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
        self.tf.summary.scalar('reward_summary', self.reward_summary)
        self.merged = tf.summary.merge_all()
        self.train_writer = tf.summary.FileWriter(self.logdir, sess.graph)

        # Trainer global
        self.global_step = 0
        self.episode = 0

    def train(self, final_steps, train=True):
        while True:
            # states are in RGBD?
            states = np.zeros((self.update_interval, 84, 84), dtype=np.uint8)
            reward = 0
            done = False
            clipped_reward = 0
            sum_of_rewards = 0
            step = 0
            state = self.env.reset()

            while True:
                if self.render:
                    self.env.render()

                state = cv2.cvtColor(state, cv2.COLOR_RGB2GRAY)
                state = cv2.resize(state, (84, 84))
                states = np.roll(states, 1, axis=0)
                states[0] = state

                if done:
                    summary, _ = self.sess.run(
                        [self.merged, self.reward_summary],
                        feed_dict={self.reward_summary: sum_of_rewards}
                    )
                    self.train_writer.add_summary(summary, self.global_step)
                    self.agent.stop_episode_and_train(
                        np.transpose(states, [1, 2, 0]), clipped_reward
                    )
                    break

                action = self.actions[
                    self.agent.act_and_train(
                        np.transpose(states, [0, 2, 0]), clipped_reward
                    )
                ]

                state, reward, done, info = self.env.step(action)
                sum_of_rewards += self.reward_clipper(reward)
                step += 1

                if train:
                    self.global_step += 1
                    if self.global_step % self.save_model_per == 0:
                        self.save_model()
            if train:
                self.episode += 1
            print('Episode: {}, Step: {}: Reward: {}'.format(
                    self.episode, self.global_step, sum_of_rewards))

            if final_steps < self.global_step:
                break

    def reward_clipper(self, reward):
        ''' clips reward (a helper function)
        '''
        if reward > 0:
            clipped_reward = 1.0
        elif reward < 0:
            clipped_reward = -1.0
        else:
            clipped_reward = 0.0

        return clipped_reward

    def save_model(self):
        ''' saves a model
        '''
        path = os.path.join(
            self.outdir, '{}/model.ckpt'.format(self.global_step)
        )
        self.saver.save(self.sess, path)
