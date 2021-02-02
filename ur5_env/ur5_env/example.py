import numpy as np
from termcolor import colored
import time
from setuptools import setup

import gym
import ur5_env

SHOW_OBS = False
RENDER = True

# env = gym.make('ur5-v0', show_obs=False, render=True)
env = gym.make('ur5-v0', render=RENDER, show_obs=SHOW_OBS, goal_mode='one', mode="normal")

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
        action = [-1.6, -1.57, 1.57, -1.57,  -1.57, 0]
        # action = [ -1.57, -1.57, 1.57, -1.57, -1.57, 0.0]

        #[-1.78003502 ,-1.11123502 , 1.72905064 ,-1.86722529, -1.42605662 , 0.08497795]
        # action = [0, -1.57, 1.57, -1.57, -1.57, 0.0]
        # action = env.action_space.sample()

        observation, reward, done, _ = env.step(action)
        if done:
            break

env.plot_actions()
env.close()

print('Finished.')
