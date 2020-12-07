import numpy as np
from cv2 import cv2
import matplotlib.pyplot as plt
from PIL import Image
from gym import utils, spaces
from gym.envs.mujoco import mujoco_env
from collections import defaultdict
from ur5_env.env.mujoco_controller import MJ_Controller
from ur5_env.env.ball_finder import Ball_Finder
import gc

# Ensure we get the path separator correct on windows
# MODEL_XML_PATH = '/home/morten/RL_husky/ur5_env/ur5_env/env/xml/UR5gripper_2_finger.xml'
MODEL_XML_PATH = '/home/morten/Documents/code/RL_husky/ur5_env/ur5_env/env/xml/UR5gripper_2_finger.xml'

class UR5(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self, reward_type='sparse', render=False, image_width=600, image_height=300, show_obs=False, mode='normal'):
        self.initialized = False
        self.mode = mode
        self.IMAGE_WIDTH = image_width
        self.IMAGE_HEIGHT = image_height
        self.rotations = {0: 0, 1: 30, 2: 60, 3: 90, 4: -30, 5: -60}
        self.step_called = 0
        utils.EzPickle.__init__(self)
        mujoco_env.MujocoEnv.__init__(self, MODEL_XML_PATH, 1)
        self.controller = MJ_Controller(self.model, self.sim, self.viewer, render)
        # self.controller = MJ_Controller()
        self.ball_finder = Ball_Finder()
        self.initialized = True
        self.show_observations = show_obs
        self.render = render
        self.distance_threshold = 0.1
        self._max_episode_steps = 1000
        # self.action_space = spaces.Box(np.array([-3.14159, -3.14159, -3.14159, -3.14159, -3.14159, -3.14159, 0.3]),
        #                                np.array([+3.14159, +3.14159, +3.14159, +3.14159, +3.14159, +3.14159, 0.3]), dtype=np.float32)
        self.action_space = spaces.Box(np.array([-3.14159, -3.14159, -3.14159, -3.14159, -3.14159, -3.14159]),
                                       np.array([+3.14159,        0, +3.14159, +3.14159, +3.14159, +3.14159]), dtype=np.float32)
        self.desired_goal = []
        self.achieved_goal = []
        self.action = [0, 0, 0, 0, 0, 0, 0]
        self.timestep_limit = 100
        self.actions_taken = 0
        self.controller.set_new_goal()

    def _set_action_space(self):
        # self.action_space = spaces.MultiDiscrete([self.IMAGE_HEIGHT*self.IMAGE_WIDTH, len(self.rotations)])
        # self.action_space = spaces.Box(np.array([-3.14159, -3.14159, -3.14159, -3.14159, -3.14159, -3.14159, 0.3]),
        #                                np.array([+3.14159, +3.14159, +3.14159, +3.14159, +3.14159, +3.14159, 0.3]), dtype=np.float32)
        self.action_space = spaces.Box(np.array([-3.14159, -3.14159, -3.14159, -3.14159, -3.14159, -3.14159]),
                                       np.array([+3.14159,        0, +3.14159, +3.14159, +3.14159, +3.14159]), dtype=np.float32)
        return self.action_space

    def step(self, action, markers=False):
        done = False
        info = {}

        if not self.initialized:
            if self.mode == 'normal':
                self.current_observation = np.zeros((9))

            elif self.mode == 'her':
                self.current_observation = defaultdict()
                self.current_observation['observation'] = np.zeros(9)
                self.current_observation['desired_goal'] = np.zeros(3)
                self.current_observation['achieved_goal'] = np.ones(3)

            else:
                print("SOMETHING WENT WRONG!")
            
            reward = -10
        else:
            self.actions_taken += 1
            # add position of grapper to move target
            action = np.append(action, [0.3])

            self.action = action

            if self.step_called == 1:
                self.current_observation = self.get_observation(show=self.show_observations)

            res = self.controller.move_group_to_joint_target(group='All', target=action, tolerance=0.1, max_steps=1000, render=self.render, quiet=False, marker=True)

            self.current_observation = self.get_observation()

            if res == 'success':
                reward = self.goal_distance(self.desired_goal, self.achieved_goal)
            else:
                reward = -10

            self.print_step_info(action, reward)

            if reward >= -0.05 or self.actions_taken >= 1:
                done = True
                self.actions_taken = 0

        self.step_called += 1

        return self.current_observation, reward, done, info

    def reset(self, show_obs=True):
        """
        Method to perform additional reset steps and return an observation.
        Gets called in the parent classes reset method.
        """

        self.controller.sim.reset()
        self.controller.set_new_goal()              
        action = [0, -1.57, 1.57, -1.57, -1.57, 0.0, 0.3]
        self.controller.move_group_to_joint_target(group='All', target=action)

        return self.get_observation(show=self.show_observations)

    def get_observation(self, show=False):
        """
        Uses the controllers get_image_data method to return an top-down image (as a np-array).
        Args:
            show: If True, displays the observation in a cv2 window.
        """

        #TODO: depth into obs space

        rgb, depth = self.controller.get_image_data(width=self.IMAGE_WIDTH, height=self.IMAGE_HEIGHT, show=show, render=self.render)
        depth = self.controller.depth_2_meters(depth)

        circle_coordinates = self.ball_finder.find_circle(rgb)
        
        observation = []
        # get joint positions
        for i in range(len(self.controller.actuated_joint_ids)):
            joints = np.append(observation, self.controller.sim.data.qpos[self.controller.actuated_joint_ids][i])

        self.desired_goal = self.controller.current_goal
        self.achieved_goal = (self.controller.sim.data.body_xpos[self.model.body_name2id('left_inner_finger')] + self.controller.sim.data.body_xpos[self.model.body_name2id('right_inner_finger')]) / 2
        
        if self.mode == 'normal':
            observation = np.append(joints, circle_coordinates)

        elif self.mode == 'her':
            observation = defaultdict()
            observation['observation'] = np.append(joints, circle_coordinates)
            observation['desired_goal'] = self.desired_goal
            observation['achieved_goal'] = self.achieved_goal

        else:
            print("SOMETHING WENT WRONG!")

        return observation

    def close(self):
        mujoco_env.MujocoEnv.close(self)
        # cv.destroyAllWindows()

    def print_step_info(self, action, reward):
        if self.mode == 'normal':
            print("######################################################################################################")
            print(f"action: \n {action}")
            print("------------------------------------------------------------------------------------------------------")
            print("observation: \n", self.current_observation[:7], "\n", self.current_observation[7:9])
            print("------------------------------------------------------------------------------------------------------")
            print(f"desired_goal: {self.desired_goal}\nachieved_goal: {self.achieved_goal}")
            print("------------------------------------------------------------------------------------------------------")
            print(f"reward: {reward}")
            print("######################################################################################################")

        elif self.mode == 'her':
            print("######################################################################################################")
            print(f"action: \n {action}")
            print("------------------------------------------------------------------------------------------------------")
            print("observation: \n", self.current_observation['observation'][:7], "\n", self.current_observation['observation'][7:9])
            print("------------------------------------------------------------------------------------------------------")
            print(f"desired_goal: {self.desired_goal}\nachieved_goal: {self.achieved_goal}")
            print("------------------------------------------------------------------------------------------------------")
            print(f"reward: {reward}")
            print("######################################################################################################")


    def print_info(self):
        # print('Model timestep:', self.model.opt.timestep)
        # print('Set number of frames skipped: ', self.frame_skip)
        # print('dt = timestep * frame_skip: ', self.dt)
        # print('Frames per second = 1/dt: ', self.metadata['video.frames_per_second'])
        # print('Actionspace: ', self.action_space)
        # print('Observation space:', self.observation_space)

        self.controller.show_model_info()

    def compute_reward(self, achieved_goal, goal, info):
        # Compute distance between goal and the achieved goal.
        d = self.goal_distance(achieved_goal, goal)
        # print(-(d > self.distance_threshold).astype(np.float32))
        return -(d > self.distance_threshold).astype(np.float32)

    def goal_distance(self, goal_a, goal_b):
        assert goal_a.shape == goal_b.shape
        sum = 0.0
        tmp = 0.0
        for i in range(3):
            tmp = goal_a[i] - goal_b[i]
            if tmp < 0:
                sum += tmp
            else:
                sum -= tmp
        return sum
