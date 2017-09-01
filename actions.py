action_space = {
    'PongDeterministic-v4': [1, 2, 3],
    'BreakoutDeterministic-v4': [1, 2, 3]
}

def get_action_space(env):
    return action_space[env]
