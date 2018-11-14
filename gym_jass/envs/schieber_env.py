import gym
from gym import spaces
from pyschieber.tournament import Tournament
from pyschieber.game import Game

from pyschieber.player.random_player import RandomPlayer
from pyschieber.player.greedy_player.greedy_player import GreedyPlayer





class SchieberEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    reward_range = (-257, 257)  # our points minus opponent team's points. reward is always given at the end of one game

    action_space = spaces.Discrete(36)  # cards on the hand of 36 cards available
    observation_space = spaces.Discrete(36)  # played cards of 36 cards available

    def __init__(self):
        tournament = Tournament(point_limit=1500)

        players = [RandomPlayer(name='Tick'), RandomPlayer(name='Trick'),
                   RandomPlayer(name='Track'), RandomPlayer(name='Dagobert')]

        [tournament.register_player(player) for player in players]

        teams = tournament.build_teams()

        game = Game(teams, 1000, False) # without counting factor to make skill more important
        game.play()

    def step(self, action):
        """

        Parameters
        ----------
        action :

        Returns
        -------
        ob, reward, episode_over, info : tuple
            ob (object) :
                an environment-specific object representing your observation of
                the environment.
            reward (float) :
                amount of reward achieved by the previous action. The scale
                varies between environments, but the goal is always to increase
                your total reward.
            episode_over (bool) :
                whether it's time to reset the environment again. Most (but not
                all) tasks are divided up into well-defined episodes, and done
                being True indicates the episode has terminated. (For example,
                perhaps the pole tipped too far, or you lost your last life.)
            info (dict) :
                 diagnostic information useful for debugging. It can sometimes
                 be useful for learning (for example, it might contain the raw
                 probabilities behind the environment's last state change).
                 However, official evaluations of your agent are not allowed to
                 use this for learning.
        """
        self._take_action(action)
        reward = self._get_reward()
        ob = self.cards_to_play
        episode_over = self.cards_to_play == 0
        return ob, reward, episode_over, {}

    def reset(self):
        self.cards_to_play = 9
        return self.cards_to_play

    def render(self, mode='human', close=False):
        print(self.cards_to_play)

    def _take_action(self, action):
        self.cards_to_play -= 1

    def _get_reward(self):
        return 1
