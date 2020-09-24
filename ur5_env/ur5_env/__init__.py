from gym.envs.registration import register

register(
    id='ur5-v0',
    entry_point='ur5_env.env:UR5',
)
