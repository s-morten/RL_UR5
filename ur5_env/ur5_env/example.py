import numpy as np
from termcolor import colored
import time
from setuptools import setup

import gym
import ur5_env

SHOW_OBS = False
RENDER = True

# env = gym.make('ur5-v0', show_obs=False, render=True)
env = gym.make('ur5-v0', render=RENDER, show_obs=SHOW_OBS)

N_EPISODES = 100
N_STEPS = 100

env.print_info()

for episode in range(1, N_EPISODES+1):
    obs = env.reset(show_obs=SHOW_OBS)
    for step in range(N_STEPS):
        print('#################################################################')
        print(colored('EPISODE {} STEP {}'.format(episode, step+1), color='white', attrs=['bold']))
        #env.controller.display_current_values()
        print('#################################################################')
        action = env.action_space.sample()
        print(action)
        # action = [100,100] # multidiscrete
        # action = 20000 #discrete
        #observation, reward, done, _ = env.step(action, record_grasps=True)
        observation, reward, done, _ = env.step(action)
        # observation, reward, done, _ = env.step(action, record_grasps=True, render=True)

env.close()

print('Finished.')
