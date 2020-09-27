import os
import numpy as np
from gym import utils, spaces
from gym.envs.mujoco import mujoco_env
from collections import defaultdict
from env.mujoco_controller import MJ_Controller

# Ensure we get the path separator correct on windows
MODEL_XML_PATH = '/home/morten/RL_husky/ur5_env/ur5_env/env/xml/UR5gripper_2_finger.xml'



class UR5(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self, reward_type='sparse', render=True, image_width=200, image_height=200):
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
        self.render=render

    def _set_action_space(self):
        self.action_space = spaces.MultiDiscrete([self.IMAGE_HEIGHT*self.IMAGE_WIDTH, len(self.rotations)])
        return self.action_space

    def step(self, action):
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

            result = self.controller.move_ee(coordinates, max_steps=1000, quiet=True, render=render, marker=markers, tolerance=0.05, plot=plot)

            self.current_observation = self.get_observation(show=self.show_observations)

            #TODO make reward
            reward = 0

        return self.current_observation, reward, done, info

    def reset(self, show_obs=True):
        """
        Method to perform additional reset steps and return an observation.
        Gets called in the parent classes reset method.
        """

        qpos = self.data.qpos
        qvel = self.data.qvel

        qpos[self.controller.actuated_joint_ids] = [0, -1.57, 1.57, -1.57, -1.57, 0.0, 0.3]

        n_objects = 40

        for i in range(n_objects):
            joint_name = f'free_joint_{i}'
            q_adr = self.model.get_joint_qpos_addr(joint_name)
            start, end = q_adr
            qpos[start] = np.random.uniform(low=-0.25, high=0.25)
            qpos[start+1] = np.random.uniform(low=-0.77, high=-0.43)
            # qpos[start+2] = 1.0
            qpos[start+2] = np.random.uniform(low=1.0, high=1.5)
            qpos[start+3:end] = Quaternion.random().unit.elements


        # n_boxes = 3
        # n_balls = 3

        # for j in ['rot', 'x', 'y', 'z']:
        #     for i in range(1,n_boxes+1):
        #         joint_name = 'box_' + str(i) + '_' + j
        #         q_adr = self.model.get_joint_qpos_addr(joint_name)
        #         if j == 'x':
        #             qpos[q_adr] = np.random.uniform(low=-0.25, high=0.25)
        #         elif j == 'y':
        #             qpos[q_adr] = np.random.uniform(low=-0.17, high=0.17)
        #         elif j == 'z':
        #             qpos[q_adr] = 0.0
        #         elif j == 'rot':
        #             start, end = q_adr
        #             qpos[start:end] = [1., 0., 0., 0.]

        #     for i in range(1,n_balls+1):
        #         joint_name = 'ball_' + str(i) + '_' + j
        #         q_adr = self.model.get_joint_qpos_addr(joint_name)
        #         if j == 'x':
        #             qpos[q_adr] = np.random.uniform(low=-0.25, high=0.25)
        #         elif j == 'y':
        #             qpos[q_adr] = np.random.uniform(low=-0.17, high=0.17)
        #         elif j == 'z':
        #             qpos[q_adr] = 0.0
        #         elif j == 'rot':
        #             start, end = q_adr
        #             qpos[start:end] = [1., 0., 0., 0.]

        self.set_state(qpos, qvel)

        self.controller.set_group_joint_target(group='All', target= qpos[self.controller.actuated_joint_ids])

        # Turn this on for training, so the objects drop down before the observation
        self.controller.stay(1000, render=self.render)
        if self.demo_mode:
            self.controller.stay(2000, render=self.render)
        # return an observation image
        return self.get_observation(show=self.show_observations)

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
