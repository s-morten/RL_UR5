import os
import numpy as np
from gym import utils, spaces
from gym.envs.mujoco import mujoco_env
from collections import defaultdict
import sys
sys.path.append('/home/morten/.local/lib/python3.6/site-packages/gym/envs/robotics')
from fetch_env import FetchEnv

# from env.robotics import fetch_env


# Ensure we get the path separator correct on windows
MODEL_XML_PATH = '/home/morten/RL_husky/ur5_env/ur5_env/env/xml/UR5gripper_2_finger.xml'



class UR5(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self, reward_type='sparse', render=True):
        mujoco_env.MujocoEnv.__init__(self, MODEL_XML_PATH, 1)
        if render:
            self.render()
        self.render=render
        utils.EzPickle.__init__(self)

    def _set_action_space(self):
        self.action_space = spaces.Discrete(6)
        return self.action_space

    def step(self, action):
        print(action)
        self.obs = defaultdict()
        self.obs['obs'] = np.zeros((5,3,6))
        return self.obs, 0, 0, 0


    def reset():
        return 0, 0, 0, 0

    def do_render(self):
        self.viewer.render()
