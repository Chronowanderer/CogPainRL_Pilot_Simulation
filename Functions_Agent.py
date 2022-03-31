import numpy as np
from Functions_Helper import *


# Valid actions without touching walls
def valid_actions(N, observation):
    agent_curr_state = np.argwhere(observation[:, :, 1] == 1)[0]
    agent_HD_state = np.argwhere(observation[:, :, 2] == 1)[0]

    ans = []
    for action in range(N):
        if agent_HD_state[0] < agent_curr_state[0]:  # Heading to North
            if action == 0:  # forward
                if observation[agent_curr_state[0] - 1, agent_curr_state[1], 0] == 0:
                    ans.append(action)
            elif action == 1:  # right
                if observation[agent_curr_state[0], agent_curr_state[1] + 1, 0] == 0:
                    ans.append(action)
            elif action == 2:  # backward
                if observation[agent_curr_state[0] + 1, agent_curr_state[1], 0] == 0:
                    ans.append(action)
            elif action == 3:  # left
                if observation[agent_curr_state[0], agent_curr_state[1] - 1, 0] == 0:
                    ans.append(action)
        elif agent_HD_state[1] > agent_curr_state[1]:  # Heading to East
            if action == 3:
                if observation[agent_curr_state[0] - 1, agent_curr_state[1], 0] == 0:
                    ans.append(action)
            elif action == 0:
                if observation[agent_curr_state[0], agent_curr_state[1] + 1, 0] == 0:
                    ans.append(action)
            elif action == 1:
                if observation[agent_curr_state[0] + 1, agent_curr_state[1], 0] == 0:
                    ans.append(action)
            elif action == 2:
                if observation[agent_curr_state[0], agent_curr_state[1] - 1, 0] == 0:
                    ans.append(action)
        elif agent_HD_state[0] > agent_curr_state[0]:  # Heading to South
            if action == 2:
                if observation[agent_curr_state[0] - 1, agent_curr_state[1], 0] == 0:
                    ans.append(action)
            elif action == 3:
                if observation[agent_curr_state[0], agent_curr_state[1] + 1, 0] == 0:
                    ans.append(action)
            elif action == 0:
                if observation[agent_curr_state[0] + 1, agent_curr_state[1], 0] == 0:
                    ans.append(action)
            elif action == 1:
                if observation[agent_curr_state[0], agent_curr_state[1] - 1, 0] == 0:
                    ans.append(action)
        elif agent_HD_state[1] < agent_curr_state[1]:  # Heading to West
            if action == 1:
                if observation[agent_curr_state[0] - 1, agent_curr_state[1], 0] == 0:
                    ans.append(action)
            elif action == 2:
                if observation[agent_curr_state[0], agent_curr_state[1] + 1, 0] == 0:
                    ans.append(action)
            elif action == 3:
                if observation[agent_curr_state[0] + 1, agent_curr_state[1], 0] == 0:
                    ans.append(action)
            elif action == 0:
                if observation[agent_curr_state[0], agent_curr_state[1] - 1, 0] == 0:
                    ans.append(action)
    return np.array(ans)


# Show current state
def current_state(observation):
    y, x = np.argwhere(observation[:, :, 1] == 1)[0]
    return (y - 1) * (observation[:, :, 1].shape[1] - 2) + (x - 1)  # relocate by excluding walls


# Shou current head direction in radius
def current_head(observation):
    agent_curr_state = np.argwhere(observation[:, :, 1] == 1)[0]
    agent_HD_state = np.argwhere(observation[:, :, 2] == 1)[0]
    if agent_HD_state[0] < agent_curr_state[0]:  # Heading to North
        return 0
    elif agent_HD_state[1] > agent_curr_state[1]:  # Heading to East
        return np.pi / 2
    elif agent_HD_state[0] > agent_curr_state[0]:  # Heading to South
        return np.pi
    elif agent_HD_state[1] < agent_curr_state[1]:  # Heading to West
        return np.pi * 3 / 2
    else:
        ValueError('Invalid state representation when calculating HD')

