from gym.envs.registration import register

register(
    id='toy-v0',
    entry_point='gym_jass.envs:ToyEnv',
)