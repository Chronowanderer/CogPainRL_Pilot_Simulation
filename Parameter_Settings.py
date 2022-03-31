from Functions_Policy import *
from Agents_Class import *


class Parameters:
    def __init__(self, set = 0):
        """
        Parameter Settings.
        """
        self._SEED = 2021 # random seed
        self._num_steps = int(1e4) # number of overall steps for training
        self._episode_steps = 50 # number of steps in a episode for training
        self._epsilon = .8  # non-guessing rate for eplison-greedy policy
        ### Model-free parameters ###
        self._step_size_Q = .2 # learning rate of Q-value
        self._discount_Q = .9  # discount of Q-value
        ### Model-based parameters ###
        self._step_size_SR = .2 # learning rate of M-value
        self._step_size_R = .2 # learning rate of R-value
        self._discount_SR = .9 # discount of M-value
        self._kappa = 1. # encoding precision of von Mises distribution
        ### Arbitration parameters ###
        self._MF_type = 'q-learn' # Model-free algorithm: ['q-learn', 'sarsa]
        self._tau = .1 # learning rate of arbitration weights
        self._alpha_mf = 1. # alpha for model-free weight
        self._alpha_mb = 1. # alpha for model-based weight
        self._beta_mf = 1. # beta for model-free weight
        self._beta_mb = 1. # beta for model-based weight
        self._rou_mf = .1 # learning rate of model-free transition rate
        self._rou_mb = .1 # learning rate of model-based transition rate

        if set == 0:
            pass
        elif set == 1:
            self._MF_type = 'sarsa'
        else:
            raise ValueError('Parameter set not implemented!')

        ### List of Policy and Agent ###
        self._Policies = {
            'random': lambda qval, obv: random_policy(qval, obv),
            'eplison': lambda qval, obv: epsilon_greedy(qval, obv, epsilon=self._epsilon, body_prior_type='uni'),
        }
        self._Agents = {
            'random': lambda qval, bhvl, eval, n: PolicyEvalAgent(qval, bhvl, eval, n,
                                                                  step_size=self._step_size_Q,
                                                                  discount=self._discount_Q),

            'sarsa': lambda qval, bhvl, eval, n: SarsaAgent(qval, bhvl, eval, n,
                                                            step_size=self._step_size_Q,
                                                            discount=self._discount_Q),

            'q-learn': lambda qval, bhvl, eval, n: QLearningAgent(qval, bhvl, eval, n,
                                                                  step_size=self._step_size_Q,
                                                                  discount=self._discount_Q),

            'sr': lambda qval, bhvl, eval, n: SuccessiveRepresentationAgent(qval, bhvl, eval, n,
                                                                            step_size_SR=self._step_size_SR,
                                                                            step_size_R=self._step_size_R,
                                                                            discount_Q=self._discount_Q,
                                                                            discount_SR=self._discount_SR,
                                                                            kappa=self._kappa),

            'mf_sr': lambda qval, bhvl, eval, n: MFandSRAgent(qval, bhvl, eval, n,
                                                              MF_type=self._MF_type,
                                                              step_size_mfQ=self._step_size_Q,
                                                              step_size_SR=self._step_size_SR,
                                                              step_size_R=self._step_size_R,
                                                              discount_mfQ=self._discount_Q,
                                                              discount_SR=self._discount_SR,
                                                              kappa=self._kappa,
                                                              tau=self._tau,
                                                              alpha_mf=self._alpha_mf,
                                                              alpha_mb=self._alpha_mb,
                                                              beta_mf=self._beta_mf,
                                                              beta_mb=self._beta_mb,
                                                              rou_mf=self._rou_mf,
                                                              rou_mb=self._rou_mb),

        }


    @property
    def seed(self):
        return self._SEED

    @property
    def Policies(self):
        return self._Policies

    @property
    def Agents(self):
        return self._Agents

    @property
    def epsilon(self):
        return self._epsilon

    @property
    def num_steps(self):
        return self._num_steps

    @property
    def episode_steps(self):
        return self._episode_steps

    @property
    def step_size_Q(self):
        return self._step_size_Q

    @property
    def discount_Q(self):
        return self._discount_Q

    @property
    def step_size_SR(self):
        return self._step_size_SR

    @property
    def step_size_R(self):
        return self._step_size_R

    @property
    def discount_SR(self):
        return self._discount_SR

    @property
    def kappa(self):
        return self._kappa

    @property
    def tau(self):
        return self._tau

    @property
    def alpha_mf(self):
        return self._alpha_mf

    @property
    def alpha_mb(self):
        return self._alpha_mb

    @property
    def beta_mf(self):
        return self._beta_mf

    @property
    def beta_mb(self):
        return self._beta_mb

    @property
    def rou_mf(self):
        return self._rou_mf

    @property
    def rou_mb(self):
        return self._rou_mb

