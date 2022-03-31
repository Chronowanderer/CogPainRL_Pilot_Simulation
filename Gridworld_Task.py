from Gridworld_Env import *


def build_gridworld_task(task,
                         discount=0.9,
                         penalty_for_shocks=-1,
                         penalty_for_walls=-1000,
                         observation_type=ObservationType.STATE_INDEX,
                         max_episode_length=200):
    """Construct a particular Gridworld layout with start/goal states.

  Args:
      task: string name of the task to use. One of {'simple', 'obstacle',
        'random_goal'}.
      discount: Discounting factor included in all Timesteps.
      penalty_for_walls: Reward added when hitting a wall (should be negative).
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
          (agent_y, agent_x, goal_y, goal_x).
      max_episode_length: If set, will terminate an episode after this many
        steps.
  """
    layout = [[-1, -1, -1, -1, -1],
              [-1, 0, 0, 0, -1],
              [-1, 0, 0, 0, -1],
              [-1, 0, 0, 0, -1],
              [-1, -1, -1, -1, -1]]
    start_state = (2, 2)  # centre
    start_head = 0  # North
    body_parts = 2  # number of electrodes
    dangerous_poss = (0.9, 0.3)
    dangerous_certainposs = (1.0, 0.0)

    tasks_specifications = {

        'body-induced-left': {
            'condition': 0,  # body-induced shocks
            'layout': layout,
            'start_state': start_state,  # centre
            'start_head': start_head,  # North
            'body_parts': body_parts,
            'dangerous_poss': dangerous_poss,
            'dangerous_direction': 3  # left
        },

        'body-induced-right': {
            'condition': 0,  # body-induced shocks
            'layout': layout,
            'start_state': start_state,  # centre
            'start_head': start_head,  # North
            'body_parts': body_parts,
            'dangerous_poss': dangerous_poss,
            'dangerous_direction': 1  # right
        },

        'world-induced-II': {
            'condition': 1,  # world-induced shocks
            'layout': layout,
            'start_state': start_state,  # centre
            'start_head': start_head,  # North
            'body_parts': body_parts,
            'dangerous_poss': dangerous_poss,
            'dangerous_centre': (2.5, 2.5)  # quadrant II
        },

        'world-induced-IV': {
            'condition': 1,  # world-induced shocks
            'layout': layout,
            'start_state': start_state,  # centre
            'start_head': start_head,  # North
            'body_parts': body_parts,
            'dangerous_poss': dangerous_poss,
            'dangerous_centre': (1.5, 1.5)  # quadrant IV
        }
    }

    return GridWorld(discount=discount,
                     penalty_for_walls=penalty_for_walls,
                     penalty_for_shocks=penalty_for_shocks,
                     observation_type=observation_type,
                     max_episode_length=max_episode_length,
                     **tasks_specifications[task])

