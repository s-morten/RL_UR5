import numpy as np
from termcolor import colored
import time
from setuptools import setup

import gym
import ur5_env

SHOW_OBS = False
RENDER = True

env = gym.make(
    "ur5-v0", render=RENDER, show_obs=SHOW_OBS, goal_mode="one", mode="normal"
)

N_EPISODES = 100000000
N_STEPS = 10

env.print_info()
action_step = [
    [2, 5, -1, -5, -0, -0],
    [2, 5, 1, -5, 0, 0],
    [2, 5, -2, -5, -0, -0],
    [0, 5, 2, -5, 0, 0],
    [0, 5, -2, -5, -0, -0],
    [-2, 5, 0, -5, -0, -0],
    [0, 5, 1, -5, -0, -0],
    [-2, 5, -1, -5, -0, -0],
    [0, 1, -5, -5, -0, -0],
    [1, 1, 2, -5, -0, 0],
    [3, 1, 2, -5, -0, -0],
    [2, 1, 1, -0, -0, -0],
    [0, 1, -2, -0, -0, -0],
    [-2, 0, 2, -0, -0, -0],
]

for episode in range(1, N_EPISODES + 1):
    env.reset()
    for step in range(N_STEPS):
        print("#################################################################")
        print(
            colored(
                "EPISODE {} STEP {}".format(episode, step + 1),
                color="white",
                attrs=["bold"],
            )
        )
        # env.controller.display_current_values()
        print("#################################################################")
        for i in range(len(action_step)):
            action = action_step[i]

            observation, reward, done, _ = env.step(action)
            if done:
                break
        if done:
            break

env.plot_actions()
env.close()

print("Finished.")
