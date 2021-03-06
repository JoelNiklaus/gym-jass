from gym.envs.registration import register

register(
    id='Toy-v0',
    entry_point='gym_jass.envs:ToyEnv',
)

register(
    id='Schieber-v0',
    entry_point='gym_jass.envs:SchieberEnv',
    kwargs={'reward_function': 'play', 'trumps': 'all', 'partner_level': 'greedy', 'opponents_level': 'greedy'},
)
