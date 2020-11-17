import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from gym import utils, spaces
from gym.envs.mujoco import mujoco_env
from collections import defaultdict
from ur5_env.env.mujoco_controller import MJ_Controller

# Ensure we get the path separator correct on windows
MODEL_XML_PATH = '/home/morten/RL_husky/ur5_env/ur5_env/env/xml/UR5gripper_2_finger.xml'


def goal_distance(goal_a, goal_b):
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


class UR5(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self, reward_type='sparse', render=False, image_width=600, image_height=300, show_obs=False, demo=False):
        self.initialized = False
        self.action_0_joint = []
        self.action_1_joint = []
        self.action_2_joint = []
        self.action_3_joint = []
        self.action_4_joint = []
        self.action_5_joint = []
        self.IMAGE_WIDTH = image_width
        self.IMAGE_HEIGHT = image_height
        self.rotations = {0: 0, 1: 30, 2: 60, 3: 90, 4: -30, 5: -60}
        self.step_called = 0
        utils.EzPickle.__init__(self)
        mujoco_env.MujocoEnv.__init__(self, MODEL_XML_PATH, 1)
        self.mode = 'her'

        self.controller = MJ_Controller(self.model, self.sim, self.viewer)
        # self.controller = MJ_Controller()
        self.initialized = True
        self.show_observations = show_obs
        self.demo_mode = demo
        self.render = render
        self.distance_threshold = 0.1
        self._max_episode_steps = 5000
        self.action_space = spaces.Box(np.array([-3.14159, -3.14159, -3.14159, -3.14159, -3.14159, -3.14159, 0.3]),
                                       np.array([+3.14159, +3.14159, +3.14159, +3.14159, +3.14159, +3.14159, 0.3]), dtype=np.float32)
        # self.action_space = spaces.Box(np.array([-3.14159, -3.14159, -3.14159, -3.14159, -3.14159, -3.14159, 0.3]),
        #                                np.array([+3.14159,        0, +3.14159, +3.14159, +3.14159, +3.14159, 0.3]), dtype=np.float32)
        self.desired_goal = []
        self.achieved_goal = []
        self.action = [0, 0, 0, 0, 0, 0, 0]

    def find_circle(self, image_array):
        # get image
        img = Image.fromarray(image_array)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        circles = cv2.HoughCircles(gray,cv2.HOUGH_GRADIENT,1,20,param1=50,param2=30,minRadius=0,maxRadius=0)
        if circles is not None:
            # convert the (x, y) coordinates and radius of the circles to integers
            circles = np.round(circles[0, :]).astype("int")
            # loop over the (x, y) coordinates and radius of the circles
            return circles[0][0], circles[0][1]
        else:
            return -1


    def _set_action_space(self):
        # self.action_space = spaces.MultiDiscrete([self.IMAGE_HEIGHT*self.IMAGE_WIDTH, len(self.rotations)])
        self.action_space = spaces.Box(np.array([-3.14159, -3.14159, -3.14159, -3.14159, -3.14159, -3.14159, 0.3]),
                                       np.array([+3.14159, +3.14159, +3.14159, +3.14159, +3.14159, +3.14159, 0.3]), dtype=np.float32)
        # self.action_space = spaces.Box(np.array([-3.14159, -3.14159, -3.14159, -3.14159, -3.14159]),
        #                                np.array([+3.14159,        0, +3.14159, +3.14159, +3.14159]), dtype=np.float32)
        return self.action_space

    def step(self, action, markers=False):
        done = False
        info = {}
        self.action_0_joint.append(action[0])
        self.action_1_joint.append(action[1])
        self.action_2_joint.append(action[2])
        self.action_3_joint.append(action[3])
        self.action_4_joint.append(action[4])
        self.action_5_joint.append(action[5])

        if not self.initialized:
            # self.current_observation = np.zeros((200,200,4))
            # self.current_observation = defaultdict()
            # self.current_observation['rgb'] = np.zeros((self.IMAGE_WIDTH,self.IMAGE_HEIGHT,3))
            # self.current_observation['depth'] = np.zeros((self.IMAGE_WIDTH,self.IMAGE_HEIGHT))
            # self.current_observation = spaces.Box(np.array(np.zeros(10)), np.array(np.ones(10)))

            # self.current_observation = defaultdict()
            # self.current_observation['observation'] = np.zeros((self.IMAGE_WIDTH,self.IMAGE_HEIGHT,3))
            # self.current_observation['desired_goal'] = np.zeros(5)
            # self.current_observation['achieved_goal'] = np.ones(5)

            # self.current_observation = np.zeros((self.IMAGE_WIDTH,self.IMAGE_HEIGHT,3))
            self.current_observation = np.zeros((24007))
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

            self.action = action

            if self.step_called == 1:
                self.current_observation = self.get_observation(show=self.show_observations)

            # qpos[self.controller.actuated_joint_ids] = [action[0], action[1], action[2], action[3], action[4], action[5], 0.3]

            # ?? self.set_state(qpos, qvel)

            # self.controller.set_group_joint_target(group='All', target= qpos[self.controller.actuated_joint_ids])

            # self.controller.set_group_joint_target(group='Arm', target=joint_angles)
            self.controller.set_new_goal(self.data.qpos)

            # self.controller.add_marker(coordinates=, label=True)
            # res = self.controller.move_group_to_joint_target(group='Arm', target=action, tolerance=0.1, max_steps=2000, render=self.render, quiet=False, marker=True)
            res = self.controller.move_group_to_joint_target(group='All', target=action, tolerance=0.1, max_steps=5000, render=self.render, quiet=False, marker=True)

            # self.current_observation = self.get_observation(show=self.show_observations)
            # print("###ACTION:###")
            # print(action)
            # print("###WE ARE:###")
            # for i in range(len(self.controller.actuators)):
            #     print('{}: P: {}, I: {}, D: {}, setpoint: {}, sample_time: {}'.format(self.controller.actuators[i][3], self.controller.actuators[i][4].tunings[0], self.controller.actuators[i][4].tunings[1],
            #                                                                     self.controller.actuators[i][4].tunings[2], self.controller.actuators[i][4].setpoint, self.controller.actuators[i][4].sample_time))

            self.current_observation = self.get_observation()

            # reward = 0
            # for i in range(3):
            #     if not math.isclose(self.current_observation['desired_goal'][i], self.current_observation['achieved_goal'][i], rel_tol=1e-1):
            #         reward = -1
            if res == 'success':
                reward = goal_distance(self.desired_goal, self.achieved_goal)
            else:
                reward = -10
            # reward = goal_distance(self.current_observation['desired_goal'], self.current_observation['achieved_goal'])
            print("######################################################################################################")
            print(f"action: {action}")
            print("------------------------------------------------------------------------------------------------------")
            # print(f"desired_goal: {self.current_observation['desired_goal']}, achieved_goal: {self.current_observation['achieved_goal']}")
            print(f"desired_goal: {self.desired_goal}, achieved_goal: {self.achieved_goal}")
            print("------------------------------------------------------------------------------------------------------")
            print(f"reward: {reward}")
            print("######################################################################################################")
            #  #  #  # self.reset() #  #  #  #
        self.step_called += 1
        # print(self.current_observation, reward, done, info)
        return self.current_observation, reward, done, info

    def reset(self, show_obs=True):
        """
        Method to perform additional reset steps and return an observation.
        Gets called in the parent classes reset method.
        """

        # qpos = self.data.qpos
        # qvel = self.data.qvel

        # qpos[self.controller.actuated_joint_ids] = [0, -1.57, 1.57, -1.57, -1.57, 0.0, 0.3]
        # action = [0, -1.57, 1.57, -1.57, -1.57, 0.0, 0.3]
        action = [0, -1.57, 1.57, -1.57, -1.57, 0.0, 0.3]
        # n_objects = 40

        # self.set_state(qpos, qvel)

        # self.controller.set_group_joint_target(group='All', target= qpos[self.controller.actuated_joint_ids])
        self.controller.move_group_to_joint_target(group='All', target=action)

        # Turn this on for training, so the objects drop down before the observation
        # self.controller.stay(1000, render=self.render)
        # if self.demo_mode:
        #    self.controller.stay(2000, render=self.render)

        # return an observation image
        return self.get_observation(show=self.show_observations)

    def get_observation(self, show=False):
        """
        Uses the controllers get_image_data method to return an top-down image (as a np-array).
        Args:
            show: If True, displays the observation in a cv2 window.
        """

        argb, depth = self.controller.get_image_data(width=self.IMAGE_WIDTH, height=self.IMAGE_HEIGHT, show=show, render=self.render)
        depth = self.controller.depth_2_meters(depth)
        # observation = defaultdict()
        # observation['rgb'] = rgb
        # observation['depth'] = depth

        # how to combine rgb and depth? is it working like this?
        # obs = np.append(rgb, depth)

        coordinates = self.find_circle(argb)

        # argb = argb[70:230, :, :]
        # argb = np.dot(argb[..., :3], [0.2989, 0.5870, 0.1140])
        #
        # img = Image.fromarray(argb)
        # img.thumbnail((300, 80), Image.ANTIALIAS)
        # argb = img.getdata()
        # argb = np.array(argb)
        # print(argb.shape)

        # img = Image.fromarray(argb, 'RGB')
        # img.save('my.png')
        # box = (0,70,600,230)
        # cropped_img = img.crop(box)
        # cropped_img.save("your.png")
        #
        # img = Image.fromarray(argb)
        # img.show()
        observation = self.action
        observation = np.append(observation, coordinates)
        # print(observation.shape)
        self.desired_goal = self.controller.current_goal
        self.achieved_goal = (self.controller.sim.data.body_xpos[self.model.body_name2id('left_inner_knuckle')] + self.controller.sim.data.body_xpos[self.model.body_name2id('right_inner_knuckle')] + self.controller.sim.data.body_xpos[self.model.body_name2id('left_inner_finger')] + self.controller.sim.data.body_xpos[self.model.body_name2id('right_inner_finger')]) / 4 + [0, 0.6, -0.36]
        # self.controller.set_goal_range(self.achieved_goal)
        # observation = defaultdict()
        # observation['observation'] = observation
        # observation['desired_goal'] = self.controller.current_goal
        # observation['achieved_goal'] = (self.controller.sim.data.body_xpos[self.model.body_name2id('left_inner_knuckle')] + self.controller.sim.data.body_xpos[self.model.body_name2id('right_inner_knuckle')] + self.controller.sim.data.body_xpos[self.model.body_name2id('left_inner_finger')] + self.controller.sim.data.body_xpos[self.model.body_name2id('right_inner_finger')]) / 4 + [0, 0.6, -0.36]

        # #observation['achieved_goal'] = (self.controller.sim.data.body_xpos[self.controller.model.body_name2id('left_inner_knuckle')] + [0, 0.588, -0.3769])
        # # self.controller.set_goal_range(observation['achieved_goal'])
        # print(self.controller.sim.data.body_xpos[self.model.body_name2id('ee_link')] - [0, -0.005, 0.16])
        # print(self.controller.sim.data.body_xpos[self.controller.model.body_name2id('wrist_3_link')])
        # print(self.controller.sim.data.body_xpos[self.controller.model.body_name2id('ee_link'))
        # print(self.controller.sim.data.body_xpos[self.model.body_name2id('base_link')]) # - [0, -0.005, 0.16]
        # print("ee_link:")
        # print(self.controller.sim.data.body_xpos[self.model.body_name2id('ee_link')])
        # print("left_inner_knuckle:")
        # print(self.controller.sim.data.body_xpos[self.model.body_name2id('left_inner_knuckle')])
        # print(self.controller.sim.data.body_xpos[self.model.body_name2id('right_inner_knuckle')])
        # print(self.controller.sim.data.body_xpos[self.model.body_name2id('left_inner_finger')])
        # print(self.controller.sim.data.body_xpos[self.model.body_name2id('right_inner_finger')])
        # print((self.controller.sim.data.body_xpos[self.model.body_name2id('left_inner_knuckle')] + self.controller.sim.data.body_xpos[self.model.body_name2id('right_inner_knuckle')] + self.controller.sim.data.body_xpos[self.model.body_name2id('left_inner_finger')] + self.controller.sim.data.body_xpos[self.model.body_name2id('right_inner_finger')]) / 4)
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

    def compute_reward(self, achieved_goal, goal, info):
        # Compute distance between goal and the achieved goal.
        d = goal_distance(achieved_goal, goal)
        print(-(d > self.distance_threshold).astype(np.float32))
        return -(d > self.distance_threshold).astype(np.float32)

    def plot_actions(self):
        plt.plot(self.action_0_joint)
        plt.plot(self.action_1_joint)
        plt.plot(self.action_2_joint)
        plt.plot(self.action_3_joint)
        plt.plot(self.action_4_joint)
        plt.plot(self.action_5_joint)
        plt.show()
