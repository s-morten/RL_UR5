import gym
import ur5_env

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common import make_vec_env
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2

# multiprocess environment
env = gym.make("ur5-v0", goal_mode="two")
model = PPO2(
    MlpPolicy,
    env,
    verbose=1,
    tensorboard_log="/home/morten/log_2_100k_lr001_steps128/log",
    n_steps=128,
    learning_rate=0.001,
)
model.learn(total_timesteps=100000)
model.save("/home/morten/log_2_100k_lr001_steps128/model")
