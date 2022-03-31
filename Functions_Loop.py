import time
import itertools
import acme
import numpy as np
from acme.utils import loggers

import pandas as pd
from Gridworld_Env import *


# @title Run loop  { form-width: "30%" }
# @markdown This function runs an agent in the environment for a number of @markdown episodes, allowing it to learn.


def run_loop(
    environment,
    agent,
    num_episodes=None,
    num_steps=None,
    logger_time_delta=1.,
    label='training_loop',
    log_loss=False,
):
    """Perform the run loop.

  We are following the Acme run loop.

  Run the environment loop for `num_episodes` episodes. Each episode is itself
  a loop which interacts first with the environment to get an observation and
  then give that observation to the agent in order to retrieve an action. Upon
  termination of an episode a new episode will be started. If the number of
  episodes is not given then this will interact with the environment
  infinitely.

  Args:
    environment: dm_env used to generate trajectories.
    agent: acme.Actor for selecting actions in the run loop.
    num_steps: number of steps to run the loop for. If `None` (default), runs
      without limit.
    num_episodes: number of episodes to run the loop for. If `None` (default),
      runs without limit.
    logger_time_delta: time interval (in seconds) between consecutive logging
      steps.
    label: optional label used at logging steps.
  """
    logger = loggers.TerminalLogger(label=label, time_delta=logger_time_delta)
    iterator = range(num_episodes) if num_episodes else itertools.count()
    all_returns = []

    num_total_steps = 0
    for episode in iterator:
        # Reset any counts and start the environment.
        start_time = time.time()
        episode_steps = 0
        episode_avoidance_action = 0
        episode_return = np.zeros(environment.body_parts) # body parts as length of pain signal vector
        episode_td_error = {'Q': [], 'SR': [], 'R': [], 'mfQ': []}
        episode_arb_rate = {'MF': [], 'MB': []}
        episode_loss = 0

        timestep = environment.reset()

        # Make the first observation.
        agent.observe_first(timestep)

        # Run an episode.
        while not timestep.last():
            # Generate an action from the agent's policy and step the environment.
            action = agent.select_action(timestep.observation)
            timestep = environment.step(action)

            # Have the agent observe the timestep and let the agent update itself.
            agent.observe(action, next_timestep=timestep)
            agent.update()

            # Book-keeping.
            episode_steps += 1
            num_total_steps += 1
            episode_return += timestep.reward
            episode_avoidance_action += environment.avoidance_action
            for key in agent.td_errors.keys():
                episode_td_error[key].append(agent.td_errors[key])
            if 'mfQ' in agent.td_errors.keys(): # Arbitration model
                for key in agent.w_arb_rate.keys():
                    episode_arb_rate[key].append(agent.w_arb_rate[key])

            if log_loss:
                episode_loss += agent.last_loss

            if num_steps is not None and num_total_steps >= num_steps:
                break

        # Collect the results and combine with counts.
        steps_per_second = episode_steps / (time.time() - start_time)
        result = {
            'episode': episode,
            'episode_length': episode_steps,
            'episode_return': episode_return,
            'episode_PAP': 1. * episode_avoidance_action / episode_steps
        }
        # episode_td
        for key in episode_td_error.keys():
            if not len(episode_td_error[key]):
                continue
            y = np.zeros((len(episode_td_error[key]), len(episode_td_error[key][0])))
            for i in range(y.shape[0]):
                y[i, :] = episode_td_error[key][i]
            result['episode_td_' + key] = np.average(y, axis=0)
        # episode_arb_rate
        for key in episode_arb_rate.keys():
            result['episode_arb_rate_' + key] = np.average(episode_arb_rate[key], axis=0)
        # loss_avg
        if log_loss:
            result['loss_avg'] = episode_loss / episode_steps

        all_returns.append(result)

        # Log the given results.
        logger.write(result)

        if num_steps is not None and num_total_steps >= num_steps:
            break

    df = pd.DataFrame(all_returns)
    df.set_index('episode', inplace=True)
    return df


# @title Implement the evaluation loop { form-width: "30%" }
# @markdown This function runs the agent in the environment for a number of episodes, without allowing it to learn, in order to evaluate it.


def evaluate(environment: dm_env.Environment, agent: acme.Actor,
             evaluation_episodes: int):
    frames = []

    for episode in range(evaluation_episodes):
        timestep = environment.reset()
        episode_return = np.zeros(environment.body_parts) # body parts as length of pain signal vector
        steps = 0
        while not timestep.last():
            frames.append(environment.plot_state(return_rgb=True))

            action = agent.select_action(timestep.observation)
            timestep = environment.step(action)
            steps += 1
            episode_return += timestep.reward
        print(
            f'Episode {episode} ended with reward {episode_return} in {steps} steps'
        )
    return frames

