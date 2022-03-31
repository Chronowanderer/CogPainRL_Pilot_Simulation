import numpy as np
from Functions_Agent import *


# Uniform random policy
def random_policy(q: np.ndarray, observation: np.ndarray):
    # return a random valid action (restricion from walls) as integer
    action = np.random.choice(valid_actions(np.shape(q)[-1], observation))

    # return a random action (no restriction from walls) as integer
    # action = np.random.randint(np.shape(q)[-1])

    return action


# Eplison greedy policy
def epsilon_greedy(
        q_values_at_bs: np.ndarray,  # Q-values at state s0: Q(b, s0, a).
        observation: np.ndarray,  # Grid observation over states.
        epsilon: float = 0.1,  # Probability of taking a random action.
        body_prior_type: str = 'uniform'  # Type of prior distribution on body parts for Q-value.
):
    # Get the number of actions & body parts from the size of the given vector of Q-values.
    num_actions = np.array(q_values_at_bs).shape[-1]
    num_bodys = np.array(q_values_at_bs).shape[0]

    # Integrate Q-values over body parts b
    weights = prior_function(num_bodys, body_prior_type)
    q_values_at_s = np.average(q_values_at_bs, axis=0, weights=weights)

    # Generate a uniform random number and compare it to epsilon to decide if the action should be greedy or not
    if epsilon < np.random.random():
        # Greedy: Pick action with the largest Q-value.
        q_max = np.NINF
        for i in valid_actions(num_actions, observation):
            if q_max < q_values_at_s[i]:
                action, q_max = i, q_values_at_s[i]
    else:
        # Else return a random action
        action = np.random.choice(valid_actions(num_actions, observation))
        # action = np.random.randint(num_actions)

    return action


def softmax_policy(
        q_values_at_bs: np.ndarray,  # Q-values at state s0: Q(b, s0, a).
        observation: np.ndarray,  # Grid observation over states.
        tau_softmax: float = 1.,  # Temperature parameter for action selection.
        body_prior_type: str = 'uniform'  # Type of prior distribution on body parts for Q-value.
):
    # Get the number of actions & body parts from the size of the given vector of Q-values.
    num_actions = np.array(q_values_at_bs).shape[-1]
    num_bodys = np.array(q_values_at_bs).shape[0]

    # Integrate Q-values over body parts b to be on (s, a)
    weights = prior_function(num_bodys, body_prior_type)
    q_values_at_s = np.average(q_values_at_bs, axis=0, weights=weights)

    # Softmax policy on action a
    pi_action = np.exp(tau_softmax * q_values_at_s)
    for a in range(num_actions):
        if a not in valid_actions(num_actions, observation):
            pi_action[a] = 0
    pi_action = pi_action / np.sum(pi_action)

    # Calculate action likelihood
    log_likelihood = 0
    for a in range(num_actions):
        if pi_action[a] > 0:
            log_likelihood += np.log(pi_action[a])

    # Select action based on softmax probability distribution
    action = np.random.choice(np.arange(num_actions), p=pi_action)
    return action
