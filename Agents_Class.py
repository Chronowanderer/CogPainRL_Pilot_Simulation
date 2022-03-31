import acme
import numpy as np

from scipy.stats import vonmises
from typing import Callable

from Gridworld_Env import *
from Functions_Agent import *


# A value-based policy takes the Q-values at a state and returns an action.
ValueBasedPolicy = Callable[[np.ndarray], int]


# @title Random Agent
class RandomAgent(acme.Actor):

    def __init__(self, environment_spec):
        """Gets the number of available actions from the environment spec."""
        self._num_actions = environment_spec.actions.num_values

    def select_action(self, observation):
        """Selects an action uniformly at random."""
        # return a random valid action (restricion from walls) as integer
        action = np.random.choice(valid_actions(self._num_actions, observation))
        # action = np.random.randint(self._num_actions)

        return action

    def observe_first(self, timestep):
        pass

    def observe(self, action, next_timestep):
        pass

    def update(self):
        pass


# @title Model-free: Policy Evaluation on Agent
class PolicyEvalAgent(acme.Actor):

    def __init__(self,
                 environment_spec,
                 behaviour_policy,
                 evaluated_policy,
                 body_parts,
                 step_size,
                 discount):

        self._state = None
        # Get number of states and actions from the environment spec.
        self._number_of_actions = environment_spec.actions.num_values
        # self._number_of_states = environment_spec.observations.num_values
        y, x = environment_spec.observations.shape[0:2]
        self._number_of_states = (x - 2) * (y - 2) # excluding walls
        self._body_parts = body_parts
        self._step_size = step_size
        self._discount = discount
        self._behaviour_policy = behaviour_policy
        self._evaluated_policy = evaluated_policy
        self._q = np.zeros((self._body_parts, self._number_of_states, self._number_of_actions))
        self._action = None
        self._next_state = None

    @property
    def q_values(self):
        return self._q

    @property
    def td_errors(self):
        return {'Q': self._td_error}

    def select_action(self, observation):
        # Select an action
        return self._behaviour_policy(self._q[:, current_state(observation), :], observation)

    def observe_first(self, timestep):
        self._state = timestep.observation

    def observe(self, action, next_timestep):
        s = self._state
        a = action
        r = next_timestep.reward
        next_s = next_timestep.observation

        # Compute TD-Error.
        self._action = a
        self._next_state = next_s
        next_a = self._evaluated_policy(self._q[:, current_state(next_s), :], next_s)
        self._td_error = r + self._discount * self._q[:, current_state(next_s), next_a] - self._q[:, current_state(s), a]

    def update(self):
        # Updates
        s = self._state
        a = self._action
        # Q-value table update.
        self._q[:, current_state(s), a] += self._step_size * self._td_error
        # Update the state
        self._state = self._next_state


# @title Model-free: SARSA Agent
class SarsaAgent(acme.Actor):

    def __init__(self,
                 environment_spec: specs.EnvironmentSpec,
                 behaviour_policy: ValueBasedPolicy,
                 evaluated_policy: ValueBasedPolicy,
                 body_parts: int,
                 step_size: float,
                 discount: float):

        # Get number of states and actions from the environment spec.
        # self._num_states = environment_spec.observations.num_values
        y, x = environment_spec.observations.shape[0:2]
        self._num_states = (x - 2) * (y - 2) # excluding walls
        self._num_actions = environment_spec.actions.num_values
        self._body_parts = body_parts

        # Create the table of Q-values, all initialized at zero.
        self._q = np.zeros((self._body_parts, self._num_states, self._num_actions))

        # Store algorithm hyper-parameters.
        self._step_size = step_size
        self._discount = discount

        # Store behavior policy.
        self._behaviour_policy = behaviour_policy

        # Containers you may find useful.
        self._state = None
        self._action = None
        self._next_state = None

    @property
    def q_values(self):
        return self._q

    @property
    def td_errors(self):
        return {'Q': self._td_error}

    def select_action(self, observation):
        return self._behaviour_policy(self._q[:, current_state(observation), :], observation)
        # return epsilon_greedy(self._q[:, current_state(observation), :], observation, self._epsilon)

    def observe_first(self, timestep):
        # Set current state.
        self._state = timestep.observation

    def observe(self, action, next_timestep):
        # Unpacking the timestep to lighten notation.
        s = self._state
        a = action
        r = next_timestep.reward
        next_s = next_timestep.observation
        # Compute the action that would be taken from the next state.
        next_a = self.select_action(next_s)
        # Compute the on-policy Q-value update.
        self._action = a
        self._next_state = next_s
        # Compute the temporal difference error
        self._td_error = r + self._discount * self._q[:, current_state(next_s), next_a] - self._q[:, current_state(s), a]

    def update(self):
        # Optional unpacking to lighten notation.
        s = self._state
        a = self._action
        # Update the Q-value table value at (s, a).
        self._q[:, current_state(s), a] += self._step_size * self._td_error
        # Update the current state.
        self._state = self._next_state


# @title Model-free: Q-Learning Agent
class QLearningAgent(acme.Actor):

    def __init__(self,
                 environment_spec: specs.EnvironmentSpec,
                 behaviour_policy: ValueBasedPolicy,
                 evaluated_policy: ValueBasedPolicy,
                 body_parts: int,
                 step_size: float,
                 discount: float):

        # Get number of states and actions from the environment spec.
        # self._num_states = environment_spec.observations.num_values
        y, x = environment_spec.observations.shape[0:2]
        self._num_states = (x - 2) * (y - 2) # excluding walls
        self._num_actions = environment_spec.actions.num_values
        self._body_parts = body_parts

        # Create the table of Q-values, all initialized at zero.
        self._q = np.zeros((self._body_parts, self._num_states, self._num_actions))

        # Store algorithm hyper-parameters.
        self._step_size = step_size
        self._discount = discount

        # Store behavior policy.
        self._behaviour_policy = behaviour_policy

        # Containers you may find useful.
        self._state = None
        self._action = None
        self._next_state = None

    @property
    def q_values(self):
        return self._q

    @property
    def td_errors(self):
        return {'Q': self._td_error}

    def select_action(self, observation):
        return self._behaviour_policy(self._q[:, current_state(observation), :], observation)

    def observe_first(self, timestep):
        # Set current state.
        self._state = timestep.observation

    def observe(self, action, next_timestep):
        # Unpacking the timestep to lighten notation.
        s = self._state
        a = action
        r = next_timestep.reward
        next_s = next_timestep.observation

        # Compute the TD error.
        self._action = a
        self._next_state = next_s
        # Compute the temporal difference error
        self._td_error = r + self._discount * np.max(self._q[:, current_state(next_s), :], axis = 1) - self._q[:, current_state(s), a]

    def update(self):
        # Optional unpacking to lighten notation.
        s = self._state
        a = self._action
        # Update the Q-value table value at (b, s, a).
        self._q[:, current_state(s), a] += self._step_size * self._td_error
        # Update the current state.
        self._state = self._next_state


# @title Model-based: Successive Representation Agent
class SuccessiveRepresentationAgent(acme.Actor):

    def __init__(self,
                 environment_spec: specs.EnvironmentSpec,
                 behaviour_policy: ValueBasedPolicy,
                 evaluated_policy: ValueBasedPolicy,
                 body_parts: int,
                 step_size_SR: float,
                 step_size_R: float,
                 discount_Q: float,
                 discount_SR: float,
                 kappa: float):

        # Get number of states and actions from the environment spec.
        # self._num_states = environment_spec.observations.num_values
        y, x = environment_spec.observations.shape[0:2]
        self._num_states = (x - 2) * (y - 2) # excluding walls
        self._num_actions = environment_spec.actions.num_values
        self._body_parts = body_parts

        # Initialize tables
        self._q = np.zeros((self._body_parts, self._num_states, self._num_actions)) # Q-values
        self._m = np.zeros((self._num_states, self._num_states)) # M-values
        for i in range(self._num_states): self._m[i, i] = 1 # Initialise M-values as identity matrix
        self._r = np.zeros((self._body_parts, self._num_states)) # R-values

        # Store algorithm hyper-parameters.
        self._step_size_SR = step_size_SR
        self._step_size_R = step_size_R
        self._discount_Q = discount_Q
        self._discount_SR = discount_SR
        self._kappa = kappa

        # Store behaviour policy.
        self._behaviour_policy = behaviour_policy

        # Containers you may find useful.
        self._state = None
        self._action = None
        self._next_state = None

        # Compute angular transit function F for calculating inner function
        delta_phi = np.pi * 2 / self._body_parts
        self._F = np.zeros(self._body_parts)
        for b in range(self._body_parts):
            theta_1 = angular_standard((b + 0.5) * delta_phi)
            theta_2 = angular_standard((b - 0.5) * delta_phi)
            if theta_1 >= theta_2:
                self._F[b] = vonmises.cdf(theta_1, self._kappa) - vonmises.cdf(theta_2, self._kappa)
            else:
                self._F[b] = vonmises.cdf(theta_1, self._kappa) - vonmises.cdf(0, self._kappa) + vonmises.cdf(2 * np.pi, self._kappa) - vonmises.cdf(theta_2, self._kappa)

    @property
    def q_values(self):
        return self._q

    @property
    def m_values(self):
        return self._m

    @property
    def r_values(self):
        return self._r

    @property
    def F_values(self):
        return self._F

    @property
    def td_errors(self):
        return {'SR': self._td_error_SR, 'R': self._td_error_R}

    def select_action(self, observation):
        return self._behaviour_policy(self._q[:, current_state(observation), :], observation)

    def observe_first(self, timestep):
        # Set current state.
        self._state = timestep.observation

    def observe(self, action, next_timestep):
        # Unpacking the timestep to lighten notation.
        s = self._state
        a = action
        r = next_timestep.reward
        g = next_timestep.discount
        next_s = next_timestep.observation
        # Compute the action that would be taken from the next state.
        next_a = self.select_action(next_s)
        # Compute the on-policy Q-value update.
        self._action = a
        self._next_state = next_s
        self._next_reward = r
        # Compute TD errors
        self._td_error_SR = self._discount_SR * self._m[current_state(self._next_state), :] - self._m[current_state(s), :] # Array over states s'
        self._td_error_SR[current_state(s)] += 1
        self._td_error_R = np.zeros(self._body_parts)
        for b in range(self._body_parts):
            self._td_error_R[b] = r[b] - np.dot(np.roll(self._F, b), self._r[:, current_state(s)])
            # self._td_error_R[b] = r[b] - self._r[b, current_state(s)]

    def update(self):
        # Optional unpacking to lighten notation.
        s = self._state
        a = self._action
        # Update table values.
        self._m[current_state(s), :] += self._step_size_SR * self._td_error_SR # M-value at (s, s').
        self._r[:, current_state(s)] += self._step_size_R * self._td_error_R # R-value at (b, s).
        # Q-value at (b, s, a).
        for b in range(self._body_parts):
            self._q[b, current_state(s), a] = self._next_reward[b] + self._discount_Q * np.dot(self._m[current_state(self._next_state), :], self._r[b, :])
        # Update the current state.
        self._state = self._next_state


# @title Arbitrated: MF-SR Agent
class MFandSRAgent(acme.Actor):

    def __init__(self,
                 environment_spec: specs.EnvironmentSpec,
                 behaviour_policy: ValueBasedPolicy,
                 evaluated_policy: ValueBasedPolicy,
                 body_parts: int,
                 step_size_mfQ: float,
                 step_size_SR: float,
                 step_size_R: float,
                 discount_mfQ: float,
                 discount_SR: float,
                 kappa: float,
                 tau: float,
                 alpha_mf: float,
                 alpha_mb: float,
                 beta_mf: float,
                 beta_mb: float,
                 rou_mf: float,
                 rou_mb: float,
                 MF_type: str = 'q-learn'
                 ):

        # Get number of states and actions from the environment spec.
        # self._num_states = environment_spec.observations.num_values
        y, x = environment_spec.observations.shape[0:2]
        self._num_states = (x - 2) * (y - 2) # excluding walls
        self._num_actions = environment_spec.actions.num_values
        self._body_parts = body_parts

        # Initialize tables
        self._q_mf = np.zeros((self._body_parts, self._num_states, self._num_actions)) # Model-free Q-values
        self._q_mb = np.zeros((self._body_parts, self._num_states, self._num_actions)) # Model-based Q-values
        self._q = np.zeros((self._body_parts, self._num_states, self._num_actions)) # Arbitrated Q-values
        self._m = np.zeros((self._num_states, self._num_states)) # M-values
        for i in range(self._num_states): self._m[i, i] = 1  # Initialise M-values as identity matrix
        self._r = np.zeros((self._body_parts, self._num_states)) # R-values

        # Store algorithm hyper-parameters.
        self._step_size_mfQ = step_size_mfQ
        self._step_size_SR = step_size_SR
        self._step_size_R = step_size_R
        self._discount_mfQ = discount_mfQ
        self._discount_SR = discount_SR
        self._kappa = kappa
        self._tau = tau
        self._alpha_mf = alpha_mf
        self._alpha_mb = alpha_mb
        self._beta_mf = beta_mf
        self._beta_mb = beta_mb
        self._rou_mf = rou_mf
        self._rou_mb = rou_mb
        self._MF_type = MF_type

        # Initialize parameters
        self._w_mb = .5 # We calculate mb-omega instead of mf-omega as weights here.
        self._w_mf = 1 - self._w_mb # Always fixed
        self._chi_mf = 0 # MF reliability
        self._chi_mb = 0 # MB reliability
        self._td_error_max_mfQ = .4 # Presumed maximum TD error of MF
        self._td_error_max_SR = .1 # Presumed maximum TD error of MB

        # Store behaviour policy.
        self._behaviour_policy = behaviour_policy

        # Containers you may find useful.
        self._state = None
        self._action = None
        self._next_state = None

        # Compute angular transit function F for calculating inner function
        delta_phi = np.pi * 2 / self._body_parts
        self._F = np.zeros(self._body_parts)
        for b in range(self._body_parts):
            theta_1 = angular_standard((b + 0.5) * delta_phi)
            theta_2 = angular_standard((b - 0.5) * delta_phi)
            if theta_1 >= theta_2:
                self._F[b] = vonmises.cdf(theta_1, self._kappa) - vonmises.cdf(theta_2, self._kappa)
            else:
                self._F[b] = vonmises.cdf(theta_1, self._kappa) - vonmises.cdf(0, self._kappa) + vonmises.cdf(2 * np.pi, self._kappa) - vonmises.cdf(theta_2, self._kappa)

    @property
    def q_values(self):
        return self._q

    @property
    def q_mf_values(self):
        return self._q_mf

    @property
    def q_mb_values(self):
        return self._q_mb

    @property
    def m_values(self):
        return self._m

    @property
    def r_values(self):
        return self._r

    @property
    def F_values(self):
        return self._F

    @property
    def td_errors(self):
        return {'mfQ': self._td_error_mfQ, 'SR': self._td_error_SR, 'R': self._td_error_R}

    @property
    def w_arb_rate(self):
        return {'MF': self._w_mf, 'MB': self._w_mb}

    def select_action(self, observation):
        return self._behaviour_policy(self._q[:, current_state(observation), :], observation)

    def observe_first(self, timestep):
        # Set current state.
        self._state = timestep.observation

    def observe(self, action, next_timestep):
        # Unpacking the timestep to lighten notation.
        s = self._state
        a = action
        r = next_timestep.reward
        g = next_timestep.discount
        next_s = next_timestep.observation
        # Compute the action that would be taken from the next state.
        next_a = self.select_action(next_s)
        # Compute the on-policy Q-value update.
        self._action = a
        self._next_state = next_s
        self._next_reward = r
        # Compute TD errors
        if self._MF_type == 'sarsa':
            self._td_error_mfQ = r + self._discount_mfQ * self._q_mf[:, current_state(next_s), next_a] - self._q_mf[:, current_state(s), a]
        elif self._MF_type == 'q-learn':
            self._td_error_mfQ = r + self._discount_mfQ * np.max(self._q_mf[:, current_state(next_s), :], axis=1) - self._q_mf[:, current_state(s), a]
        else:
            raise ValueError('Invaild model-free algorithm assigned.')
        self._td_error_SR = self._discount_SR * self._m[current_state(self._next_state), :] - self._m[current_state(s), :] # Array over states s'
        self._td_error_SR[current_state(s)] += 1
        self._td_error_R = np.zeros(self._body_parts)
        for b in range(self._body_parts):
            self._td_error_R[b] = r[b] - np.dot(np.roll(self._F, b), self._r[:, current_state(s)])
            # self._td_error_R[b] = r[b] - self._r[b, current_state(s)]
        # Compute arbitration rates as weights
        self._ape_predict_error_mf = np.abs(np.mean(self._td_error_mfQ)) # ?????
        self._ape_predict_error_mb = np.abs(np.mean(self._td_error_SR)) # ?????
        self._chi_mf += self._rou_mf * (1. - self._chi_mf - self._ape_predict_error_mf / self._td_error_max_mfQ)
        self._chi_mb += self._rou_mb * (1. - self._chi_mb - self._ape_predict_error_mb / self._td_error_max_SR)
        self._w_mb += self._tau * (self._w_mf * self._alpha_mf / (1. + np.exp(self._beta_mf) * self._chi_mf) - self._w_mb * self._alpha_mb / (1. + np.exp(self._beta_mb) * self._chi_mb))
        self._w_mf = 1 - self._w_mb

    def update(self):
        # Optional unpacking to lighten notation.
        s = self._state
        a = self._action
        # Update table values.
        self._m[current_state(s), :] += self._step_size_SR * self._td_error_SR # M-value at (s, s').
        self._r[:, current_state(s)] += self._step_size_R * self._td_error_R # R-value at (b, s).
        # Q-values at (b, s, a).
        self._q_mf[:, current_state(s), a] += self._step_size_mfQ * self._td_error_mfQ
        for b in range(self._body_parts):
            self._q_mb[b, current_state(s), a] = self._next_reward[b] + self._discount_mfQ * np.dot(self._m[current_state(self._next_state), :], self._r[b, :])
        self._q = self._w_mb * self._q_mb + self._w_mf * self._q_mf
        # Update the current state.
        self._state = self._next_state

