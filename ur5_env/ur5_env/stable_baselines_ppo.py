import gym
import ur5_env

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common import make_vec_env
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2

# multiprocess environment
# env = make_vec_env('ur5-v0')
env = DummyVecEnv([lambda: gym.make('ur5-v0')])
# model = PPO2(MlpPolicy, env, verbose=1, gamma=0, learning_rate=0.001, tensorboard_log='./log_mode_one')
model = PPO2.load("ppo2_model_less_as", env=env, verbose=1, gamma=0, learning_rate=0.001, tensorboard_log='./log2')
# model.set_env(env)
model.learn(total_timesteps=100000)
model.save("ppo2_model_2")
