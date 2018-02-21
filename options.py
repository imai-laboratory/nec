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
        elif self.environment == 'CartPole-v0':
            self.model = CARTPOLE_MODEL

        self.convs = self.models['cnns']
        self.fcs = self.models['fcs']

        # FOR AGENT LEARNING
        self.rep_buffer_size = 10 ** 5
        self.batch_size = 32
        self.learning_starts = 10000
        self.gamma = 0.99
        self.n_step = 100  # N-Step DQN
        self.train_freq = 16
        self.learning_starts = 10000

        # Build Train
        self.lr = 2.5e-4
        self.momentum = 0.95
        self.epsilon = 1e-2

        # DND OPTIONS
        self.hin_size = 512
        self.grad_norm_clipping = 10.0

        self.num_actions = None
