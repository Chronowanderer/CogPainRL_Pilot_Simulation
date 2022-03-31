#!/usr/bin/env python
# coding: utf-8


# @title Import modules
from Parameter_Settings import *
from Gridworld_Run import *


# @title Task Setting
Parameters_Set = 0
Task_Names = ['body-induced-left', 'body-induced-right', 'world-induced-II', 'world-induced-IV']
Agent_Names = ['random', 'sarsa', 'q-learn', 'sr', 'mf_sr'] # 'random' for random agent
Policy_Names = ['random', 'eplison'] # 'random' for random agent


# @title Simulation
if __name__ == "__main__":
    System_Settings(Parameters = Parameters(Parameters_Set))
    Simulation(task_name = Task_Names[3],
               agent_name = Agent_Names[3],
               policy_name = Policy_Names[1],
               Parameters = Parameters(Parameters_Set))
