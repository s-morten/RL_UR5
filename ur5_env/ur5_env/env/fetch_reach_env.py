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
import math

MODEL_XML_PATH = "xml/UR5gripper_2_finger.xml"


class UR5(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(
        self,
        reward_type="sparse",
        render=False,
        image_width=600,
        image_height=300,
        show_obs=False,
        mode="normal",
        goal_mode="nine",
    ):
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
        self.action_space = spaces.Box(
            np.array([-5.0, -5.0, -5.0, -5.0, -5.0, -5.0]),
            np.array([+5.0, +5.0, +5.0, +5.0, +5.0, +5.0]),
            dtype=np.float32,
        )
        self.desired_goal = []
        self.achieved_goal = []
        self.action = [0, 0, 0, 0, 0, 0, 0]
        self.timestep_limit = 100
        self.actions_taken = 0
        self.goal_mode = goal_mode
        self.controller.set_new_goal(mode=self.goal_mode)
        self.coordinate_x = 0
        self.coordinate_y = 0
        self.coordinate_depth = 0

    def _set_action_space(self):
        self.action_space = spaces.Box(
            np.array([-5.0, -5.0, -5.0, -5.0, -5.0, -5.0]),
            np.array([+5.0, +5.0, +5.0, +5.0, +5.0, +5.0]),
            dtype=np.float32,
        )
        return self.action_space

    def step(self, action, markers=False):
        done = False
        info = {}

        if not self.initialized:
            if self.mode == "normal":
                observation = np.zeros((10))
                self.current_observation = np.zeros((10))

            elif self.mode == "her":
                self.current_observation = defaultdict()
                self.current_observation["observation"] = np.zeros(10)
                self.current_observation["desired_goal"] = np.zeros(3)
                self.current_observation["achieved_goal"] = np.ones(3)

            else:
                print("SOMETHING WENT WRONG!")

            reward = -10
        else:
            all_actions_allowed = True
            self.actions_taken += 1
            # add position of grapper to move target
            state = self.current_observation
            deg_state = []
            for j in range(6):
                deg_state = np.append(deg_state, math.degrees(state[j]))

            action_buffer = []
            for i in range(6):
                tmp = math.radians(deg_state[i] + action[i])
                if tmp > math.pi or tmp < -math.pi:
                    all_actions_allowed = False
                    if tmp > math.pi:
                        tmp = math.pi
                    else:
                        tmp = -math.pi
                    print("unallowed action, lol!")
                action_buffer = np.append(action_buffer, tmp)
            if all_actions_allowed:
                action = np.append(action_buffer, [0.3])
            self.print_to_file(action=action)

            self.action = action

            if self.step_called == 1:
                observation = self.get_observation(show=self.show_observations)

            joints = []
            res_step_one = "success"
            res_step_two = self.controller.move_group_to_joint_target(
                group="All",
                target=action,
                tolerance=0.01,
                max_steps=1000,
                render=self.render,
                quiet=False,
                marker=True,
            )
            observation = self.get_observation()

            if (
                res_step_one == "success"
                and res_step_two == "success"
                and all_actions_allowed
            ):
                reward_dis = self.goal_distance(self.desired_goal, self.achieved_goal)
                if self.mode == "normal":
                    alpha = abs(self.current_observation[1])
                    beta = abs(self.current_observation[2])
                    gamma = abs(self.current_observation[3])
                elif self.mode == "her":
                    alpha = abs(self.current_observation["observation"][1])
                    beta = abs(self.current_observation["observation"][2])
                    gamma = abs(self.current_observation["observation"][3])

                if self.current_observation[0] < 0 and self.current_observation[2] > 0:
                    reward_add_on = 0 - abs(self.current_observation[4] + 0.5 * math.pi)
                elif (
                    self.current_observation[0] > 0 and self.current_observation[2] < 0
                ):
                    reward_add_on = 0 - abs(self.current_observation[4] - 0.5 * math.pi)
                else:
                    reward_add_on = -25
                # reward_add_on = 0

                reward_ang = 0.5 * math.pi - (
                    2 * math.pi - alpha - (math.pi - beta) - gamma
                )

                if reward_dis <= 2:
                    reward = (
                        -0.5 * math.log2(reward_dis)
                        + 0.5
                        - abs(reward_ang)
                        - abs(reward_add_on)
                    )
                else:
                    reward = (
                        -0.5 * reward_dis + 1 - abs(reward_ang) - abs(reward_add_on)
                    )

                if reward_dis <= 0.05:
                    reward = 10 - abs(reward_add_on) - abs(reward_ang)
                    if reward >= 9.85:
                        reward = 20
                        done = True
            else:
                reward = -100
                reward_dis = 100
                reward_ang = 0
                reward_add_on = 0

            # if reward >= -0.01 or self.actions_taken >= 1:
            if self.actions_taken >= 17:
                done = True
                self.actions_taken = 0

            if self.mode == "her":
                if reward >= -0.05:
                    reward = 1
                else:
                    reward = 0
            self.print_to_reward(reward=reward, distance=reward_dis)
            self.print_step_info(action, reward, reward_dis, reward_ang, reward_add_on)

        self.step_called += 1
        print("step called: ", self.step_called)

        return observation, reward, done, info

    def reset(self, show_obs=True):
        """
        Method to perform additional reset steps and return an observation.
        Gets called in the parent classes reset method.
        """

        self.controller.sim.reset()

        self.controller.set_new_goal(mode=self.goal_mode)
        action = [-1.57, -1.57, 1.57, -1.57, -1.57, 0.0, 0.3]
        self.controller.move_group_to_joint_target(group="All", target=action)

        observation = self.get_observation(
            show=self.show_observations, coordinates=True
        )

        return observation

    def get_observation(self, show=False, coordinates=True):
        """
        Uses the controllers get_image_data method to return an top-down image (as a np-array).
        Args:
            show: If True, displays the observation in a cv2 window.
        """
        if coordinates:
            rgb, depth = self.controller.get_image_data(
                width=self.IMAGE_WIDTH,
                height=self.IMAGE_HEIGHT,
                show=show,
                render=self.render,
            )
            depth = self.controller.depth_2_meters(depth)

            circle_coordinates = self.ball_finder.find_circle(rgb)

            depth_val = depth[circle_coordinates[0]][circle_coordinates[1]]

            self.coordinate_x = circle_coordinates[0]
            self.coordinate_y = circle_coordinates[1]

            self.coordinate_depth = depth_val

        else:
            depth_val = self.coordinate_depth
            circle_coordinates = np.array(self.coordinate_x, self.coordinate_y)

        joints = []
        # get joint positions
        for i in range(len(self.controller.actuated_joint_ids)):
            joints = np.append(
                joints,
                self.controller.sim.data.qpos[self.controller.actuated_joint_ids][i],
            )

        self.desired_goal = self.controller.current_goal
        self.achieved_goal = (
            self.controller.sim.data.body_xpos[
                self.model.body_name2id("left_inner_finger")
            ]
            + self.controller.sim.data.body_xpos[
                self.model.body_name2id("right_inner_finger")
            ]
        ) / 2

        if self.mode == "normal":
            observation = np.append(joints, circle_coordinates)
            observation = np.append(observation, depth_val)

        elif self.mode == "her":
            observation = defaultdict()
            observation["observation"] = np.append(joints, circle_coordinates)
            observation["observation"] = np.append(
                observation["observation"], depth_val
            )
            observation["desired_goal"] = self.desired_goal
            observation["achieved_goal"] = self.achieved_goal

        else:
            print("SOMETHING WENT WRONG!")

        self.print_to_file(state=joints)

        self.current_observation = observation

        return observation

    def close(self):
        mujoco_env.MujocoEnv.close(self)
        # cv.destroyAllWindows()

    def print_step_info(self, action, reward, reward_dis, reward_ang, reward_add_on):
        if self.mode == "normal":
            print(
                "######################################################################################################"
            )
            print(f"action: \n {action}")
            print(
                "------------------------------------------------------------------------------------------------------"
            )
            print(
                "observation: \n",
                self.current_observation[:7],
                "\n",
                self.current_observation[7:9],
                "\n",
                self.current_observation[9:10],
            )
            print(
                "------------------------------------------------------------------------------------------------------"
            )
            print(
                f"desired_goal: {self.desired_goal}\nachieved_goal: {self.achieved_goal}"
            )
            print(
                "------------------------------------------------------------------------------------------------------"
            )
            print(
                f"reward: {reward}; reward_distance: {reward_dis}, reward_angel: {reward_ang}, reward_add_on: {reward_add_on}"
            )
            print(
                "######################################################################################################"
            )

        elif self.mode == "her":
            print(
                "######################################################################################################"
            )
            print(f"action: \n {action}")
            print(
                "------------------------------------------------------------------------------------------------------"
            )
            print(
                "observation: \n",
                self.current_observation["observation"][:7],
                "\n",
                self.current_observation["observation"][7:9],
            )
            print(
                "------------------------------------------------------------------------------------------------------"
            )
            print(
                f"desired_goal: {self.desired_goal}\nachieved_goal: {self.achieved_goal}"
            )
            print(
                "------------------------------------------------------------------------------------------------------"
            )
            print(
                f"reward: {reward}; reward_distance: {reward_dis}, reward_angel: {reward_ang}, reward_add_on: {reward_add_on}"
            )
            print(
                "######################################################################################################"
            )

    def print_info(self):

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
            sum += tmp * tmp
        sum = math.sqrt(sum)
        return sum

    def print_to_file(self, action=None, state=None):
        f = open("/home/morten/log_2_100k_lr001_steps128/plot.txt", "a")
        if state is not None:
            for i in range(7):
                f.write(f"O{i}: {str(state[i])}\n")
        if action is not None:
            for i in range(7):
                f.write(f"A{i}: {str(action[i])}\n")
        f.close()

    def print_to_reward(self, reward=0, distance=0):
        f = open("/home/morten/log_2_100k_lr001_steps128/reward.txt", "a")
        f.write(f"R: {str(reward)}\n")
        f.write(f"D: {str(distance)}\n")
        f.close()

    def render(self):
        print("render")
        print(self.action)
