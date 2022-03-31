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
        q_values_at_bs: np.ndarray,  # Q-values in state s: Q(b, s, a).
        observation: np.ndarray,  # Grid observation over states.
        epsilon: float = 0.1,  # Probability of taking a random action.
        body_prior_type: str = 'uniform'  # Type of prior distribution on body parts for Q-value.
):
    """Return an epsilon-greedy action sample."""
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
