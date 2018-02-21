import os
CARTPOLE_MODEL = {
    'cnns': [],
    'fcs': [64, 64]
}

PONG_MODEL = {
    'cnns': [(32, 8, 4), (64, 4, 2), (64, 3, 1)],
    'fcs': [512]
}


class Options:
    def __init__(self, args):

        # ARGS OPTIONS
        self.environment = args.env
        self.target_update_interval = args.target_update_interval
        self.update_interval = args.update_interval
        self.final_exploration_frames = args.final_exploration_frames
        self.final_steps = args.final_steps
        self.replay_start_size = args.replay_start_size

        self.render = args.render
        self.gpu = args.gpu
        self.load = args.load

        # Create Working Directories
        # absolute outdir
        self.outdir = os.path.join(
            os.path.dirname(__file__), 'results_' + args.outdir)
        if not os.path.exists(self.outdir):
            os.makedirs(self.outdir)
        # absolute logdir
        self.logdir = os.path.join(
            os.path.dirname(__file__), 'logs/' + args.logdir)

        # Neural Networks
        if self.environment == 'PongDeterministic-v4':
            self.model = PONG_MODEL
        elif self.environment == 'CartPole-v1':
            self.model = CARTPOLE_MODEL

        self.convs = self.model['cnns']
        self.fcs = self.model['fcs']
        self.encoded_size = self.fcs[-1]

        # FOR AGENT LEARNING
        self.rep_buffer_size = 10 ** 5
        self.batch_size = 32
        # self.learning_starts = 10000
        self.learning_starts = 500
        self.gamma = 0.99
        self.n_step = 100  # N-Step DQN
        self.train_freq = 16
        self.learning_starts = 10000

        # Build Train
        # self.lr = 2.5e-4
        self.lr = 10e-2
        self.momentum = 0.95
        # self.epsilon = 10e-2
        self.epsilon = 10e-2

        # DND OPTIONS
        self.hin_size = self.encoded_size
        self.grad_norm_clipping = 10.0
        self.capacity = 10 ** 3
        self.p = 10

        # TODO: magic
        self.num_actions = 2

        # TODO: remove
        self.device = '\gpu:0' if self.gpu else '\cpu:0'

        # TODO: remove
        # self.in_shape = (1, 84, 84, self.update_interval)
        self.in_shape = (1, 4, self.update_interval)
        self.image_size = ()

        self.profile = False
