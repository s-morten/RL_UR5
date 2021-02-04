import gym
import ur5_env

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common import make_vec_env
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2

# multiprocess environment
env = gym.make('ur5-v0', goal_mode='random')
# env = make_vec_env('ur5-v0')
# env = DummyVecEnv([lambda: gym.make('ur5-v0')])
model = PPO2(MlpPolicy, env, verbose=1,tensorboard_log='./log_random_new_step_norm_param_300000')
# model = PPO2.load("ppo2_model_less_as", env=env, verbose=1, gamma=0, learning_rate=0.001, tensorboard_log='./log2')
# model.set_env(env)
model.learn(total_timesteps=300000)
model.save("ppo2_model_random_300000_new_step_norm_param")
