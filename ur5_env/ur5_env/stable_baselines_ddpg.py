import gym
import ur5_env
import numpy as np

from stable_baselines.ddpg.policies import MlpPolicy
from stable_baselines.common.noise import (
    NormalActionNoise,
    OrnsteinUhlenbeckActionNoise,
    AdaptiveParamNoiseSpec,
)
from stable_baselines import DDPG

env = gym.make("ur5-v0")

# the noise objects for DDPG
n_actions = env.action_space.shape[-1]
param_noise = None
action_noise = OrnsteinUhlenbeckActionNoise(
    mean=np.zeros(n_actions), sigma=float(0.5) * np.ones(n_actions)
)

model = DDPG(
    MlpPolicy,
    env,
    verbose=1,
    param_noise=param_noise,
    action_noise=action_noise,
    tensorboard_log="./log_ddpg",
)
model.learn(total_timesteps=200000)
model.save("ddpg_100000")
