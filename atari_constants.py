REPLAY_BUFFER_SIZE = 10 ** 5
BATCH_SIZE = 32
LEARNING_START_STEP = 10 ** 4
FINAL_STEP = 10 ** 7
GAMMA = 0.99
N_STEP = 100
UPDATE_INTERVAL = 16
STATE_WINDOW = 4
EXPLORATION_DURATION = 10 ** 6
CONVS = [(32, 8, 4), (64, 4, 2), (64, 3, 1)]
FCS = []

LR = 2.5e-4
MOMENTUM = 0.95
EPSILON = 1e-2
GRAD_CLIPPING = 10.0

DND_CAPACITY = 5 * 10 ** 5
DND_P = 50
DND_KEY_SIZE = 512

DEVICE = '/gpu:0'
