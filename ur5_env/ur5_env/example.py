import numpy as np
from termcolor import colored
import time
from setuptools import setup

import gym
import ur5_env

SHOW_OBS = False
RENDER = False

# env = gym.make('ur5-v0', show_obs=False, render=True)
env = gym.make('ur5-v0', render=RENDER, show_obs=SHOW_OBS)

N_EPISODES = 100000000
N_STEPS = 200

env.print_info()

for episode in range(1, N_EPISODES+1):
    env.reset()
    for step in range(N_STEPS):
        print('#################################################################')
        print(colored('EPISODE {} STEP {}'.format(episode, step+1), color='white', attrs=['bold']))
        #env.controller.display_current_values()
        print('#################################################################')
        # action = [0, -1.57, 1.57, -1.57, -1.57, 0.0, 0.3]
        # action = [0, -3.14, 1.57, -1.57, -1.57, 0.0, 0.3]
        # All
        # -3.14 / 0
        #
        action = env.action_space.sample()
        # action = [100,100] # multidiscrete
        # action = 20000 #discrete
        # observation, reward, done, _ = env.step(action, record_grasps=True)
        observation, reward, done, _ = env.step(action)
        if done:
            break
        # observation, reward, done, _ = env.step(action, record_grasps=True, render=True)
        # obs = env.reset(show_obs=SHOW_OBS)
env.plot_actions()
env.close()

print('Finished.')
