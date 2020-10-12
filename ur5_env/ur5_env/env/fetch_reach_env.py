import os
import numpy as np
from gym import utils, spaces
from gym.envs.mujoco import mujoco_env
from collections import defaultdict
from ur5_env.env.mujoco_controller import MJ_Controller

# Ensure we get the path separator correct on windows
MODEL_XML_PATH = '/home/morten/RL_husky/ur5_env/ur5_env/env/xml/UR5gripper_2_finger.xml'



class UR5(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self, reward_type='sparse', render=False, image_width=200, image_height=200, show_obs=True, demo=False):
        self.initialized = False
        self.IMAGE_WIDTH = image_width
        self.IMAGE_HEIGHT = image_height
        self.rotations = {0: 0, 1: 30, 2: 60, 3: 90, 4: -30, 5: -60}
        self.step_called = 0
        utils.EzPickle.__init__(self)
        mujoco_env.MujocoEnv.__init__(self, MODEL_XML_PATH, 1)
        # if render:
        #     self.render()

        self.controller = MJ_Controller(self.model, self.sim, self.viewer)
        #self.controller = MJ_Controller()
        self.initialized = True
        self.show_observations = show_obs
        self.demo_mode = demo
        self.render=render
        self._set_action_space()

    def _set_action_space(self):
        # self.action_space = spaces.MultiDiscrete([self.IMAGE_HEIGHT*self.IMAGE_WIDTH, len(self.rotations)])
        self.action_space = spaces.Box( np.array([-3.14159,-3.14159,-3.14159,-3.14159,-3.14159]), np.array([+3.14159,0,+3.14159,+3.14159,+3.14159]), dtype=np.float32)
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
            # x = action[0] % self.IMAGE_WIDTH
            # y = action[0] // self.IMAGE_WIDTH
            # rotation = action[1]
            #
            # depth = self.current_observation['depth'][y][x]
            #
            # coordinates = self.controller.pixel_2_world(pixel_x=x, pixel_y=y, depth=depth, height=self.IMAGE_HEIGHT, width=self.IMAGE_WIDTH)
            #
            # result = self.controller.move_ee(coordinates, max_steps=1000, quiet=True, render=self.render, marker=markers, tolerance=0.05)

            if self.step_called == 1:
                self.current_observation = self.get_observation(show=self.show_observations)

            # qpos[self.controller.actuated_joint_ids] = [action[0], action[1], action[2], action[3], action[4], action[5], 0.3]

            # ?? self.set_state(qpos, qvel)

            # self.controller.set_group_joint_target(group='All', target= qpos[self.controller.actuated_joint_ids])

            #self.controller.set_group_joint_target(group='Arm', target=joint_angles)

            # self.controller.add_marker(coordinates=, label=True)
            self.controller.get_new_goal()
            self.controller.move_group_to_joint_target(group='Arm', target=action, tolerance=0.1, max_steps=1000, render=self.render, quiet=False, marker=True)

            # self.current_observation = self.get_observation(show=self.show_observations)

            self.current_observation = self.get_observation()

            #TODO make reward
            reward = 0
        self.step_called += 1

        return self.current_observation, reward, done, info

    def reset(self, show_obs=True):
        """
        Method to perform additional reset steps and return an observation.
        Gets called in the parent classes reset method.
        """

        # qpos = self.data.qpos
        # qvel = self.data.qvel

        #qpos[self.controller.actuated_joint_ids] = [0, -1.57, 1.57, -1.57, -1.57, 0.0, 0.3]
        action = [0, -1.57, 1.57, -1.57, -1.57, 0.0, 0.3]
        #n_objects = 40

        #self.set_state(qpos, qvel)

        #self.controller.set_group_joint_target(group='All', target= qpos[self.controller.actuated_joint_ids])
        self.controller.move_group_to_joint_target(group='All', target=action)

        # Turn this on for training, so the objects drop down before the observation
        #self.controller.stay(1000, render=self.render)
        #if self.demo_mode:
        #    self.controller.stay(2000, render=self.render)

        # return an observation image
        return self.get_observation(show=self.show_observations)

    def get_observation(self, show=False):
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

    def close(self):
        mujoco_env.MujocoEnv.close(self)
        cv.destroyAllWindows()

    def print_info(self):
        # print('Model timestep:', self.model.opt.timestep)
        # print('Set number of frames skipped: ', self.frame_skip)
        # print('dt = timestep * frame_skip: ', self.dt)
        # print('Frames per second = 1/dt: ', self.metadata['video.frames_per_second'])
        # print('Actionspace: ', self.action_space)
        # print('Observation space:', self.observation_space)

        self.controller.show_model_info()
