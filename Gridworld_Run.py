import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from Functions_Device import *
from Functions_Random import *
from Functions_Plot import *
from Functions_Loop import *
from Agents_Class import *
from Gridworld_Task import *


# Random Seeds and Device
def System_Settings(Parameters, is_inline = False):
    set_seed(seed=Parameters.seed)
    print('Current Device:', set_device())

    if not is_inline:
        mpl.use('Qt5Agg')
    plt.style.use("https://raw.githubusercontent.com/NeuromatchAcademy/course-content/master/nma.mplstyle")
    mpl.rc('image', cmap='Blues')


# @title Q-value Learning
def Simulation(task_name, agent_name, policy_name, Parameters):
    grid = build_gridworld_task(task=task_name, observation_type=ObservationType.GRID, max_episode_length=Parameters.episode_steps)
    environment, environment_spec = setup_environment(grid)
    policy = Parameters.Policies[policy_name]
    agent = Parameters.Agents[agent_name](environment_spec, policy, policy, environment.body_parts)
    Run_Simulation(agent, environment, grid, Parameters)
    plt.show()


# run experiment and get the value functions from agent
def Run_Simulation(agent, environment, grid, Parameters):
    returns = run_loop(environment=environment, agent=agent, num_steps=Parameters.num_steps)
    print('AFTER {0:,} STEPS:'.format(Parameters.num_steps))

    # show pain return curves
    plot_stats(returns, condition=environment.condition)
    plt.show(block=False)

    # show Q-values
    for body_part in range(environment.body_parts + 1):
        if body_part < environment.body_parts:
            q = agent.q_values[body_part].reshape(tuple(np.shape(grid._layout) - np.array([2, 2])) + (4, ))
        elif body_part == environment.body_parts:
            weights = prior_function(environment.body_parts)
            q = np.average(agent.q_values, axis = 0, weights = weights).reshape(tuple(np.shape(grid._layout) - np.array([2, 2])) + (4, ))

        if body_part == 0:
            print('\nLEFT ARM after {0:,} STEPS ...'.format(Parameters.num_steps))
        elif body_part == 1:
            print('\nRIGHT ARM after {0:,} STEPS ...'.format(Parameters.num_steps))
        elif body_part == environment.body_parts:
            print('\nWHOLE BODY after {0:,} STEPS ...'.format(Parameters.num_steps))

        # NaN the unavailable forward actions at corners
        layout = environment.layout
        q[0, 0, 0] = np.NaN
        q[layout[0], 0, 0] = np.NaN
        q[0, layout[1], 0] = np.NaN
        q[layout[0], layout[1], 0] = np.NaN

        # plot Q-values and V-values
        plot_action_values(q, epsilon=1.)
        # grid.plot_greedy_policy(q) # Visualize the greedy policy
        plt.show(block = False)

