import numpy as np
from termcolor import colored
import time
from setuptools import setup

import gym
import ur5_env

SHOW_OBS = False
RENDER = True

N_EPISODES = 100000000
N_STEPS = 5

# env = gym.make('ur5-v0', show_obs=False, render=True)
env = gym.make('ur5-v0', render=RENDER, show_obs=SHOW_OBS)

env.print_info()

for episode in range(1, N_EPISODES+1):
    env.reset()
    for step in range(N_STEPS):
        print('#################################################################')
        print(colored('EPISODE {} STEP {}'.format(episode, step+1), color='white', attrs=['bold']))
        #env.controller.display_current_values()
        print('#################################################################')
        goal = env.desired_goal
        print(goal)
        env.controller.move_ee(goal)
# env.plot_actions()
env.close()

print('Finished.')
# create controller instance
controller = MJ_Controller()

# Display robot information
controller.show_model_info()

# Move ee to position above the object, plot the trajectory to an image file, show a marker at the target location
controller.move_ee([0.0, -0.6 , 0.95], plot=True, marker=True)

# Move down to object
controller.move_ee([0.0, -0.6 , 0.895])

# Wait a second
controller.stay(1000)

# Attempt grasp
controller.grasp()

# Move up again
controller.move_ee([0.0, -0.6 , 1.0])

# Throw the object away
controller.toss_it_from_the_ellbow()

# Wait before finishing
controller.stay(2000)
