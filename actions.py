action_space = {
    'PongDeterministic-v4': [1, 2, 3],
    'BreakoutDeterministic-v4': [1, 2, 3],
    'CartPole-v1': [0, 1]
}

# https://github.com/openai/gym/wiki/CartPole-v0

def get_action_space(env):
    return action_space[env]
