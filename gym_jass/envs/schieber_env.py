import asyncio
import logging
import threading
import time

import jsonpickle
import websockets

import gym
from gym import spaces
from schieber.card import Card, from_tuple_to_card, from_card_to_tuple, from_card_to_index, from_index_to_card, \
    from_string_to_index

from schieber.game import Game
from schieber.player.random_player import RandomPlayer
from schieber.player.greedy_player.greedy_player import GreedyPlayer
from schieber.player.external_player import ExternalPlayer
from schieber.suit import Suit
from schieber.team import Team
from schieber.tournament import Tournament

from gym_jass.envs.game_server import GameServer
from gym_jass.envs.tournament_server import TournamentServer

logger = logging.getLogger(__name__)


class SchieberEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    reward_range = (-257, 257)  # our points minus opponent team's points. reward is always given at the end of one game

    """
    Tuple and Dict spaces are not compatible with openai baselines
    suit_space = spaces.Discrete(5)  # the last one is for NO SUIT
    value_space = spaces.Discrete(10)  # the last one is for NO VALUE
    card_space = spaces.Tuple((suit_space, value_space))
    #card_space = spaces.MultiDiscrete([[0, 3], [0, 8]])
    stich_space = spaces.Tuple(
        (card_space, card_space, card_space, card_space))  # a stich which has been played always contains 4 cards
    played_stack_space = spaces.Tuple(
        (stich_space, stich_space, stich_space, stich_space, stich_space, stich_space, stich_space,
         stich_space))  # the stack of the played stichs contains 8 stichs at most
    deck_space = spaces.Tuple((card_space, card_space, card_space))  # the table contains 3 cards at most
    hand_space = spaces.Tuple((card_space, card_space, card_space, card_space, card_space, card_space, card_space,
                               card_space, card_space))  # the hand contains 9 cards at most

    action_space = card_space
    # observation_space = spaces.Tuple((played_stack_space, deck_space, hand_space))
    observation_space = hand_space
    """

    # 0 stands for NO CARD, 1 to 36 are mapped to the 36 possible cards
    # the one card to be chosen
    action_space = spaces.Discrete(36)

    # index 0 to 8: the player's hand
    # index 9 to 40: the cards in the stack which have been turned over in the order of appearance
    # index 41 to 43: the cards currently on the table: 41 --> 1st card, 42 --> 2nd card, 43 --> 3rd card
    observation_space = spaces.Box(low=0, high=36, shape=(44,), dtype=int)

    def __init__(self):
        self.action = {}
        self.observation = {}

        self.start_jass_server()

    def __del__(self):
        logger.info("Environment has been stopped.")

    def start_jass_server(self):
        self.player = ExternalPlayer(name='GYM-RL')
        players = [RandomPlayer(name='Tick', seed=1), RandomPlayer(name='Trick', seed=2),
                   RandomPlayer(name='Track', seed=3), self.player]
        self.tournament = Tournament(point_limit=1500, seed=0)
        [self.tournament.register_player(player) for player in players]

        thread = threading.Thread(target=self.tournament.play)
        thread.start()

        # action = Card(Suit.ROSE, 9)
        #
        # obs = self.player.get_observation(False)
        # print(self.tournament.teams[0].points, self.tournament.teams[1].points)
        # print(self.tournament.games[-1].cards_on_table)
        # print(self.tournament.games[-1].stiche)
        # print(obs)
        # self.player.set_action(action)
        #
        #
        # obs = self.player.get_observation()
        # print(self.tournament.teams[0].points, self.tournament.teams[1].points)
        # print(self.tournament.games[-1].cards_on_table)
        # print(self.tournament.games[-1].stiche)
        # print(obs)
        # self.player.set_action(action)
        #
        # obs = self.player.get_observation()
        # print(self.tournament.teams[0].points, self.tournament.teams[1].points)
        # print(self.tournament.games[-1].cards_on_table)
        # print(self.tournament.games[-1].stiche)
        # print(obs)
        # self.player.set_action(action)
        #
        # time.sleep(100)

        # server = TournamentServer(tournament)
        # stop = asyncio.Future()
        # return asyncio.get_event_loop().create_task(server.start(stop))

        # team_1 = Team(players=[players[0], players[2]])
        # team_2 = Team(players=[players[1], players[3]])
        # teams = [team_1, team_2]
        # game = Game(teams, point_limit=1000, use_counting_factor=False, seed=1)
        # GameServer(game)

    def step(self, action):
        """Run one timestep of the environment's dynamics. When end of
        episode is reached, you are responsible for calling `reset()`
        to reset this environment's state.

        Accepts an action and returns a tuple (observation, reward, done, info).

        Parameters
        ----------
        action (object): an action provided by the environment

        Returns
        -------
        observation, reward, episode_over, info : tuple
            observation (object) :
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
        logger.info("stepping in the environment")

        self._take_action(action)
        reward = self._get_reward()
        observation = self.observation_dict_to_index(self.observation)
        episode_over = self.observation['teams'][0]['points'] + self.observation['teams'][1]['points'] == 157
        return observation, reward, episode_over, {}

    def reset(self):
        """Resets the state of the environment and returns an initial observation.

                Returns: observation (object): the initial observation of the
                    space.
        """
        logger.info("resetting the environment")

        self.tournament.teams[0].points = 0
        self.tournament.teams[1].points = 0
        self.observation = {}

        wait = True
        if self.observation == {}:
            wait = False
        observation = self.player.get_observation(wait)
        self.observation = observation

        return self.observation_dict_to_index(observation)

    def render(self, mode='human', close=False):
        """Renders the environment.

                The set of supported modes varies per environment. (And some
                environments do not support rendering at all.) By convention,
                if mode is:

                - human: render to the current display or terminal and
                  return nothing. Usually for human consumption.
                - rgb_array: Return an numpy.ndarray with shape (x, y, 3),
                  representing RGB values for an x-by-y pixel image, suitable
                  for turning into a video.
                - ansi: Return a string (str) or StringIO.StringIO containing a
                  terminal-style text representation. The text can include newlines
                  and ANSI escape sequences (e.g. for colors).

                Note:
                    Make sure that your class's metadata 'render.modes' key includes
                      the list of supported modes. It's recommended to call super()
                      in implementations to use the functionality of this method.

                Args:
                    mode (str): the mode to render with
                    close (bool): close all open renderings

                Example:

                class MyEnv(Env):
                    metadata = {'render.modes': ['human', 'rgb_array']}

                    def render(self, mode='human'):
                        if mode == 'rgb_array':
                            return np.array(...) # return RGB frame suitable for video
                        elif mode is 'human':
                            ... # pop up a window and render
                        else:
                            super(MyEnv, self).render(mode=mode) # just raise an exception
                """
        logger.info("rendering the environment")

        print(self.observation)

    def seed(self, seed=None):
        """Sets the seed for this env's random number generator(s).

        Note:
            Some environments use multiple pseudorandom number generators.
            We want to capture all such seeds used in order to ensure that
            there aren't accidental correlations between multiple generators.

        Returns:
            list<bigint>: Returns the list of seeds used in this env's random
              number generators. The first value in the list should be the
              "main" seed, or the value which a reproducer should pass to
              'seed'. Often, the main seed equals the provided 'seed', but
              this won't be true if seed=None, for example.
        """
        logger.info("seeding the environment")

        # Seeding makes everything more predictable and reproducible.
        # But this could also be a problem because the players might be more exploitable.

        # if seed is not None:
        #    self.game.seed = seed + 0
        #    self.game.players[1].seed = seed + 1
        #    self.game.players[2].seed = seed + 2
        #    self.game.players[3].seed = seed + 3
        # return
        pass

    def close(self):
        logger.info("closing the environment")


    def _take_action(self, action):
        action += 1  # action is sampled between 0 and 35 but must be between 1 and 36!
        action = from_index_to_card(action)
        self.action = action
        self.player.set_action(action)
        self.observation = self.player.get_observation()

    def _get_reward(self):
        reward = self.observation['teams'][0]['points'] - self.observation['teams'][1]['points']
        reward = self.tournament.teams[0].points - self.tournament.teams[1].points
        return reward

    def observation_dict_to_tuple(self, observation):
        hand = [(4, 9)] * 9
        for i in range(len(observation["cards"])):
            hand[i] = from_card_to_tuple(observation["cards"][i])
        return tuple(hand)

    def observation_dict_to_index(self, observation):
        hand = [0] * 9
        for i in range(len(observation["cards"])):
            hand[i] = from_card_to_index(observation["cards"][i])

        # leave stack for simplicity for now
        stack = [0] * (8 * 4)
        for i in range(len(observation["stiche"])):
            stich = observation["stiche"][i]["played_cards"]
            for j in range(len(stich)):
                stack[4 * i + j] = from_string_to_index(stich[j]['card'])

        table = [0] * 3
        for i in range(len(observation["table"])):
            table[i] = from_string_to_index(observation["table"][i]["card"])

        return hand + stack + table
