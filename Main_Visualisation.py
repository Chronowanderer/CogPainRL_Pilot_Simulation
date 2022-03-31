# Visualise GridWorlds

import matplotlib as mpl
from Parameter_Settings import *
from Gridworld_Run import *
from Functions_Loop import *
from Agents_Class import *


# Settings
Parameters_Set = 0
System_Settings(Parameters = Parameters(Parameters_Set))


# Instantiate two tabular environments
body_induced_grid = build_gridworld_task(task='body-induced-left', observation_type=ObservationType.GRID)
world_induced_grid = build_gridworld_task(task='world-induced-IV', observation_type=ObservationType.GRID)

# Plot them
body_induced_grid.plot_grid()
plt.title('body-induced')
plt.show()
world_induced_grid.plot_grid()
plt.title('world-induced')
plt.show()


# @title Look at environment_spec

# Selecting one tabular environment
example_grid = body_induced_grid
#example_grid = world_induced_grid

# Note: setup_environment is implemented in the same cell as GridWorld.
environment, environment_spec = setup_environment(example_grid)

print('actions:\n', environment_spec.actions, '\n')
print('observations:\n', environment_spec.observations, '\n')
print('rewards:\n', environment_spec.rewards, '\n')
print('discounts:\n', environment_spec.discounts, '\n')

environment.reset()
environment.plot_state()
plt.title('start')
plt.show()


# @title Pick an action and see the state changing
action = "forward"  #@param ["forward", "right", "backward", "left"] {type:"string"}
action_int = {'forward': 0, 'right': 1, 'backward': 2, 'left': 3}
timestep = environment.step(int(action_int[action]))  # pytype: dm_env.TimeStep
environment.plot_state()
plt.title(action)
plt.show()


# @title Visualisation of a random agent in GridWorld { form-width: "30%" }
# Create the agent by giving it the action space specification.
agent = RandomAgent(environment_spec)
# Run the agent in the evaluation loop, which returns the frames.
frames = evaluate(environment, agent, evaluation_episodes=1)
# Visualize the random agent's episode.
display_video(frames)
