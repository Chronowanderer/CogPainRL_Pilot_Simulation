# Implement GridWorld { form-width: "30%" }

import enum
import dm_env
import random

import numpy as np
import matplotlib.pyplot as plt

from acme import specs
from acme import wrappers


class ObservationType(enum.IntEnum):
    STATE_INDEX = enum.auto()
    AGENT_ONEHOT = enum.auto()
    GRID = enum.auto()
    AGENT_GOAL_POS = enum.auto()


class GridWorld(dm_env.Environment):

    def __init__(
            self,
            condition,  # 0 for body-induced, 1 for world-induced
            layout,
            start_state,
            start_head,  # 0123 - NESW
            body_parts,
            dangerous_poss=(0, 0),  # (high_poss, low_poss)
            dangerous_direction=None,
            dangerous_centre=None,
            dangerous_range=(1, 1),
            observation_type=ObservationType.STATE_INDEX,
            discount=0.9,
            penalty_for_shocks=-1,
            penalty_for_walls=-1000,
            max_episode_length=None):
        """Build a grid environment.

    Simple gridworld defined by a map layout, a start and a goal state.

    Layout should be a NxN grid, containing:
      * 0: empty
      * -1: wall

    Args:
      layout: NxN array of numbers, indicating the layout of the environment.
      start_state: Tuple (y, x) of starting location.
      observation_type: Enum observation type to use. One of:
        * ObservationType.STATE_INDEX: int32 index of agent occupied tile.
        * ObservationType.AGENT_ONEHOT: NxN float32 grid, with a 1 where the
          agent is and 0 elsewhere.
        * ObservationType.GRID: NxNx4 float32 grid of feature channels.
          First channel contains walls (1 if wall, 0 otherwise),
          second the agent position (1 if agent, 0 otherwise),
          third the agent head direction (1 if agent, 0 otherwise) corresponding to the agent position,
          and fourth goal position (1 if goal, 0 otherwise)
        * ObservationType.AGENT_GOAL_POS: float32 tuple with
          (agent_y, agent_x, goal_y, goal_x)
      discount: Discounting factor included in all Timesteps.
      penalty_for_walls: Reward added when hitting a wall (should be negative).
      max_episode_length: If set, will terminate an episode after this many
        steps.
    """
        if observation_type not in ObservationType:
            raise ValueError('observation_type should be a ObservationType instace.')
        self._layout = np.array(layout)
        self._condition = condition
        self._start_state = start_state
        self._state = self._start_state
        self._start_head = start_head
        self._head = self._start_head
        self._body_parts = body_parts
        self._dangerous_poss = dangerous_poss
        self._dangerous_direction = dangerous_direction
        self._dangerous_centre = dangerous_centre
        self._dangerous_range = dangerous_range
        self._number_of_states = np.prod(np.shape(self._layout) - np.array([2, 2]))  # excluding walls
        self._discount = discount
        self._penalty_for_shocks = penalty_for_shocks
        self._penalty_for_walls = penalty_for_walls
        self._observation_type = observation_type
        self._layout_dims = self._layout.shape
        self._max_episode_length = max_episode_length
        self._num_episode_steps = 0

    @property
    def condition(self):
        return self._condition

    @property
    def layout(self):
        return self._layout

    @property
    def number_of_states(self):
        return self._number_of_states

    @property
    def start_state(self):
        return self._start_state

    @property
    def start_head(self):
        return self._start_head

    @property
    def state(self):
        return self._state

    @property
    def head(self):
        return self._head

    @property
    def body_parts(self):
        return self._body_parts

    @property
    def dangerous_poss(self):
        return self._dangerous_poss

    @property
    def dangerous_direction(self):
        return self._dangerous_direction

    @property
    def dangerous_centre(self):
        return self._dangerous_centre

    @property
    def dangerous_range(self):
        return self._dangerous_range

    @property
    def avoidance_action(self):
        return self._avoidance_action

    def set_state(self, x, y):
        self._state = (y, x)

    def set_head(self, h):
        self._head = h

    def observation_spec(self):
        if self._observation_type is ObservationType.AGENT_ONEHOT:
            return specs.Array(shape=self._layout_dims,
                               dtype=np.float32,
                               name='observation_agent_onehot')
        elif self._observation_type is ObservationType.GRID:
            return specs.Array(shape=self._layout_dims + (4,),
                               dtype=np.float32,
                               name='observation_grid')
        elif self._observation_type is ObservationType.AGENT_GOAL_POS:
            return specs.Array(shape=(4,),
                               dtype=np.float32,
                               name='observation_agent_goal_pos')
        elif self._observation_type is ObservationType.STATE_INDEX:
            return specs.DiscreteArray(self._number_of_states,
                                       dtype=int,
                                       name='observation_state_index')

    def action_spec(self):
        return specs.DiscreteArray(4, dtype=int, name='action')

    def get_obs(self):
        if self._observation_type is ObservationType.AGENT_ONEHOT:
            obs = np.zeros(self._layout.shape, dtype=np.float32)
            # Place agent
            obs[self._state] = 1
            return obs
        elif self._observation_type is ObservationType.GRID:
            obs = np.zeros(self._layout.shape + (4,), dtype=np.float32)
            # layout
            obs[..., 0] = self._layout < 0
            # agent position
            obs[self._state[0], self._state[1], 1] = 1
            # HD position corresponding to the current agent position
            if self._head == 0:  # North HD
                obs[self._state[0] - 1, self._state[1], 2] = 1
            elif self._head == 1:  # East HD
                obs[self._state[0], self._state[1] + 1, 2] = 1
            elif self._head == 2:  # South HD
                obs[self._state[0] + 1, self._state[1], 2] = 1
            elif self._head == 3:  # West HD
                obs[self._state[0], self._state[1] - 1, 2] = 1
            return obs
        elif self._observation_type is ObservationType.AGENT_GOAL_POS:
            return np.array(self._state + self._goal_state, dtype=np.float32)
        elif self._observation_type is ObservationType.STATE_INDEX:
            y, x = self._state
            return y * self._layout.shape[1] + x

    def reset(self):
        self._state = self._start_state
        self._num_episode_steps = 0
        return dm_env.TimeStep(step_type=dm_env.StepType.FIRST,
                               reward=None,
                               discount=None,
                               observation=self.get_obs())

    def step(self, action):
        y, x = self._state
        h = self._head
        self._avoidance_action = 0

        if h == 0:  # heading North
            if action == 0:  # forward
                new_state = (y - 1, x)
                new_head = 0
            elif action == 1:  # right
                new_state = (y, x + 1)
                new_head = 1
            elif action == 2:  # backward
                new_state = (y + 1, x)
                new_head = 2
            elif action == 3:  # left
                new_state = (y, x - 1)
                new_head = 3
            else:
                raise ValueError('Invalid action: {} is not 0, 1, 2, or 3.'.format(action))
        elif h == 1:  # heading East
            if action == 0:  # forward
                new_state = (y, x + 1)
                new_head = 1
            elif action == 1:  # right
                new_state = (y + 1, x)
                new_head = 2
            elif action == 2:  # backward
                new_state = (y, x - 1)
                new_head = 3
            elif action == 3:  # left
                new_state = (y - 1, x)
                new_head = 0
            else:
                raise ValueError('Invalid action: {} is not 0, 1, 2, or 3.'.format(action))
        elif h == 2:  # heading South
            if action == 0:  # forward
                new_state = (y + 1, x)
                new_head = 2
            elif action == 1:  # right
                new_state = (y, x - 1)
                new_head = 3
            elif action == 2:  # backward
                new_state = (y - 1, x)
                new_head = 0
            elif action == 3:  # left
                new_state = (y, x + 1)
                new_head = 1
            else:
                raise ValueError('Invalid action: {} is not 0, 1, 2, or 3.'.format(action))
        elif h == 3:  # heading West
            if action == 0:  # forward
                new_state = (y, x - 1)
                new_head = 3
            elif action == 1:  # right
                new_state = (y - 1, x)
                new_head = 0
            elif action == 2:  # backward
                new_state = (y, x + 1)
                new_head = 1
            elif action == 3:  # left
                new_state = (y + 1, x)
                new_head = 2
            else:
                raise ValueError('Invalid action: {} is not 0, 1, 2, or 3.'.format(action))
        new_h = new_head
        new_y, new_x = new_state

        if (self._condition == 0) and (action != self._dangerous_direction):  # body-centred
            self._avoidance_action = 1
        elif self._condition == 1:
            self._avoidance_action = 1
            if np.abs(y - self._dangerous_centre[0]) <= self._dangerous_range[0]:
                if np.abs(x - self._dangerous_centre[1]) <= self._dangerous_range[1]:
                    if np.abs(new_y - self._dangerous_centre[0]) <= self._dangerous_range[0]:
                        if np.abs(new_x - self._dangerous_centre[1]) <= self._dangerous_range[1]:
                            self._avoidance_action = 0

        step_type = dm_env.StepType.MID
        if self._layout[new_y, new_x] == -1:  # wall
            reward = self._penalty_for_walls
            reward_body = reward * np.ones(self._body_parts)
            discount = self._discount
            new_state = (y, x)
            raise ValueError('Bumping into the wall!')
        elif self._layout[new_y, new_x] == 0:  # empty cell
            reward_body = np.zeros(self._body_parts)  # [left, right] for 2 body parts
            if self._condition == 0:  # body-centred
                # shocks
                if action == self._dangerous_direction:
                    if random.uniform(0, 1) <= self._dangerous_poss[0]:
                        if self._dangerous_direction == 1:
                            reward_body[1] = self._penalty_for_shocks
                        elif self._dangerous_direction == 3:
                            reward_body[0] = self._penalty_for_shocks
                else:
                    if random.uniform(0, 1) <= self._dangerous_poss[1]:
                        if self._dangerous_direction == 1:
                            reward_body[0] = self._penalty_for_shocks
                        elif self._dangerous_direction == 3:
                            reward_body[1] = self._penalty_for_shocks

            elif self._condition == 1:  # world-centred
                # shocks
                vector = (self._dangerous_centre[0] - new_y, self._dangerous_centre[1] - new_x)
                if (abs(vector[0]) <= self._dangerous_range[0]) and (abs(vector[1]) <= self._dangerous_range[1]):
                    if random.uniform(0, 1) <= self._dangerous_poss[0]:
                        if new_h == 0:
                            if vector[1] < 0: reward_body[0] = self._penalty_for_shocks
                            if vector[1] > 0: reward_body[1] = self._penalty_for_shocks
                        if new_h == 1:
                            if vector[0] > 0: reward_body[0] = self._penalty_for_shocks
                            if vector[0] < 0: reward_body[1] = self._penalty_for_shocks
                        if new_h == 2:
                            if vector[1] > 0: reward_body[0] = self._penalty_for_shocks
                            if vector[1] < 0: reward_body[1] = self._penalty_for_shocks
                        if new_h == 3:
                            if vector[0] < 0: reward_body[0] = self._penalty_for_shocks
                            if vector[0] > 0: reward_body[1] = self._penalty_for_shocks
                else:
                    if random.uniform(0, 1) <= self._dangerous_poss[1]:
                        if new_h == 0:
                            if vector[1] < 0: reward_body[0] = self._penalty_for_shocks
                            if vector[1] > 0: reward_body[1] = self._penalty_for_shocks
                        if new_h == 1:
                            if vector[0] > 0: reward_body[0] = self._penalty_for_shocks
                            if vector[0] < 0: reward_body[1] = self._penalty_for_shocks
                        if new_h == 2:
                            if vector[1] > 0: reward_body[0] = self._penalty_for_shocks
                            if vector[1] < 0: reward_body[1] = self._penalty_for_shocks
                        if new_h == 3:
                            if vector[0] < 0: reward_body[0] = self._penalty_for_shocks
                            if vector[0] > 0: reward_body[1] = self._penalty_for_shocks

            reward = np.sum(reward_body)
            discount = self._discount
        else:  # a goal
            reward = self._layout[new_y, new_x]
            reward_body = reward * np.ones(self._body_parts)
            discount = 0.
            new_state = self._start_state
            new_head = self._start_head
            step_type = dm_env.StepType.LAST

        self._state = new_state
        self._head = new_head
        self._num_episode_steps += 1
        if (self._max_episode_length is not None
                and self._num_episode_steps >= self._max_episode_length):
            step_type = dm_env.StepType.LAST

        return dm_env.TimeStep(step_type=step_type,
                               reward=np.float32(reward_body),
                               discount=discount,
                               observation=self.get_obs())

    def plot_grid(self, add_start=True):
        plt.figure(figsize=(4, 4))
        plt.imshow(self._layout <= -1, interpolation='nearest')
        ax = plt.gca()
        ax.grid(0)
        plt.xticks([])
        plt.yticks([])
        # Add start/goal
        if add_start:
            plt.text(self._start_state[1],
                     self._start_state[0],
                     r'$\mathbf{S}$',
                     fontsize=16,
                     ha='center',
                     va='center')
        h, w = self._layout.shape
        for y in range(h - 1):
            plt.plot([-0.5, w - 0.5], [y + 0.5, y + 0.5], '-w', lw=2)
        for x in range(w - 1):
            plt.plot([x + 0.5, x + 0.5], [-0.5, h - 0.5], '-w', lw=2)

    def plot_state(self, return_rgb=False):
        self.plot_grid(add_start=False)
        # Add the agent location
        HD_signs = ['^', '>', 'v', '<']
        plt.text(
            self._state[1],
            self._state[0],
            HD_signs[self._head],
            # fontname='symbola',
            fontsize=18,
            ha='center',
            va='center',
        )
        if return_rgb:
            fig = plt.gcf()
            plt.axis('tight')
            plt.subplots_adjust(0, 0, 1, 1, 0, 0)
            fig.canvas.draw()
            data = np.fromstring(fig.canvas.tostring_rgb(),
                                 dtype=np.uint8,
                                 sep='')
            w, h = fig.canvas.get_width_height()
            data = data.reshape((h, w, 3))
            plt.close(fig)
            return data

    def plot_policy(self, policy):
        action_names = [
            r'$\uparrow$', r'$\rightarrow$', r'$\downarrow$', r'$\leftarrow$'
        ]
        self.plot_grid()
        plt.title('Policy Visualization')
        h, w = self._layout.shape

        # excluding walls
        for y in range(1, h - 1):
            for x in range(1, w - 1):
                # if (y, x) != self._goal_state:
                action_name = action_names[policy[y - 1, x - 1]]
                plt.text(x, y, action_name, ha='center', va='center')

    def plot_greedy_policy(self, q):
        greedy_actions = np.argmax(q, axis=2)
        self.plot_policy(greedy_actions)


def setup_environment(environment):
    """Returns the environment and its spec."""

    # Make sure the environment outputs single-precision floats.
    environment = wrappers.SinglePrecisionWrapper(environment)

    # Grab the spec of the environment.
    environment_spec = specs.make_environment_spec(environment)

    return environment, environment_spec

