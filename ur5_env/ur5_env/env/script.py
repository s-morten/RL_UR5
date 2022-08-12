import numpy as np
from termcolor import colored
import time
from setuptools import setup

import gym
import ur5_env
from mujoco_controller import MJ_Controller


controller = MJ_Controller(render=True)


action = [0, -1.57, 1.57, -1.57, -1.57, 0.0, 0.3]
controller.move_group_to_joint_target(group="All", target=action, render=True)
controller.set_new_goal([0.0, -0.5, 0.04])
action = [-1.83075118, -1.57, 1.92007267, -1.9957577, -1.62144184, 0.18844855, 0.3]
controller.move_group_to_joint_target(group="All", target=action, render=True)
action = [
    -1.83075118,
    -1.34453058,
    1.92007267,
    -1.9957577,
    -1.62144184,
    0.18844855,
    0.3,
]
controller.move_group_to_joint_target(group="All", target=action, render=True)

action = [0, -1.57, 1.57, -1.57, -1.57, 0.0, 0.3]
controller.move_group_to_joint_target(group="All", target=action, render=True)
controller.set_new_goal([0.25, -0.91, 0.04])
action = [-1.44489658, -1.57, 1.45630288, -1.67203045, -1.61417532, 0.30790743, 0.3]
controller.move_group_to_joint_target(group="All", target=action, render=True)
action = [
    -1.44489658,
    -0.90122318,
    1.45630288,
    -1.67203045,
    -1.61417532,
    0.30790743,
    0.3,
]
controller.move_group_to_joint_target(group="All", target=action, render=True)

action = [0, -1.57, 1.57, -1.57, -1.57, 0.0, 0.3]
controller.move_group_to_joint_target(group="All", target=action, render=True)
controller.set_new_goal([0.25, -0.5, 0.04])
action = [-1.3546164, -1.57, 2.01256967, -2.08015394, -1.44699156, -0.72324598, 0.3]
controller.move_group_to_joint_target(group="All", target=action, render=True)
action = [
    -1.3546164,
    -1.03049707,
    2.01256967,
    -2.08015394,
    -1.44699156,
    -0.72324598,
    0.3,
]
controller.move_group_to_joint_target(group="All", target=action, render=True)
