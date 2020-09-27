import os
import numpy as np
from gym import utils, spaces
from gym.envs.mujoco import mujoco_env
from collections import defaultdict
from ur5_env.env.mujoco_controller import MJ_Controller

# Ensure we get the path separator correct on windows
MODEL_XML_PATH = '/home/morten/RL_husky/ur5_env/ur5_env/env/xml/UR5gripper_2_finger.xml'



class UR5(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self, reward_type='sparse', render=True, image_width=200, image_height=200, show_obs=True):
        self.initialized = False
        self.IMAGE_WIDTH = image_width
        self.IMAGE_HEIGHT = image_height
        self.rotations = {0: 0, 1: 30, 2: 60, 3: 90, 4: -30, 5: -60}
        utils.EzPickle.__init__(self)
        mujoco_env.MujocoEnv.__init__(self, MODEL_XML_PATH, 1)
        if render:
            self.render()
        self.controller = MJ_Controller(self.model, self.sim, self.viewer)
        self.initialized = True
        self.show_observations = show_obs
        self.render=render

    def _set_action_space(self):
        self.action_space = spaces.MultiDiscrete([self.IMAGE_HEIGHT*self.IMAGE_WIDTH, len(self.rotations)])
        return self.action_space

    def step(self, action, markers=False):
        done = False
        info = {}

        if not self.initialized:
            # self.current_observation = np.zeros((200,200,4))
            self.current_observation = defaultdict()
            self.current_observation['rgb'] = np.zeros((self.IMAGE_WIDTH,self.IMAGE_HEIGHT,3))
            self.current_observation['depth'] = np.zeros((self.IMAGE_WIDTH,self.IMAGE_HEIGHT))
            reward = 0
        else:
            x = action[0] % self.IMAGE_WIDTH
            y = action[0] // self.IMAGE_WIDTH
            rotation = action[1]

            depth = self.current_observation['depth'][y][x]

            coordinates = self.controller.pixel_2_world(pixel_x=x, pixel_y=y, depth=depth, height=self.IMAGE_HEIGHT, width=self.IMAGE_WIDTH)

            result = self.controller.move_ee(coordinates, max_steps=1000, quiet=True, render=self.render, marker=markers, tolerance=0.05)

            self.current_observation = self.get_observation(show=self.show_observations)

        return self.current_observation, reward, done, info

    def reset():
        return 0, 0, 0, 0

    #def do_render(self):
    #    self.viewer.render()

    def get_observation(self, show=True):
        """
        Uses the controllers get_image_data method to return an top-down image (as a np-array).
        Args:
            show: If True, displays the observation in a cv2 window.
        """

        rgb, depth = self.controller.get_image_data(width=self.IMAGE_WIDTH, height=self.IMAGE_HEIGHT, show=show)
        depth = self.controller.depth_2_meters(depth)
        observation = defaultdict()
        observation['rgb'] = rgb
        observation['depth'] = depth

        return observation
