import logging
import threading

import gym
from gym import spaces
from schieber.card import from_card_to_tuple, from_card_to_index, from_index_to_card, from_string_to_index
from schieber.game import Game

from schieber.player.random_player import RandomPlayer
from schieber.player.external_player import ExternalPlayer
from schieber.team import Team
from schieber.tournament import Tournament

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
    # index 9 to 44: the cards in the stack which have been turned over in the order of appearance
    # index 45 to 47: the cards currently on the table: 45 --> 1st card, 46 --> 2nd card, 47 --> 3rd card
    observation_space = spaces.Box(low=0, high=36, shape=(48,), dtype=int)

    def __init__(self):
        self.action = {}
        self.observation = {}

        self.player = ExternalPlayer(name='GYM-RL')
        players = [RandomPlayer(name='Random Opponent 1', seed=1), RandomPlayer(name='Random Partner', seed=2),
                   RandomPlayer(name='Random Opponent 2', seed=3), self.player]
        # self.tournament = Tournament(point_limit=1500, seed=0)
        # [self.tournament.register_player(player) for player in players]

        team_1 = Team(players=[players[0], players[2]])
        team_2 = Team(players=[players[1], players[3]])
        self.teams = [team_1, team_2]
        self.game = Game(self.teams, point_limit=1000, use_counting_factor=False, seed=1)

        self.start_jass_server()

    def __del__(self):
        logger.info("Environment has been stopped.")

    def start_jass_server(self):
        # thread = threading.Thread(target=self.tournament.play)
        # thread.start()

        thread = threading.Thread(target=self.game.play_endless)
        thread.start()

        #
        # action = Card(Suit.ROSE, 9)
        #
        # obs = self.player.get_observation(False)
        # print(self.teams[0].points, self.teams[1].points)
        # print(self.game.cards_on_table)
        # print(self.game.stiche)
        # print(obs)
        # self.player.set_action(action)
        #
        #
        # obs = self.player.get_observation()
        # print(self.teams[0].points, self.teams[1].points)
        # print(self.game.cards_on_table)
        # print(self.game.stiche)
        # print(obs)
        # self.player.set_action(action)
        #
        # obs = self.player.get_observation()
        # print(self.teams[0].points, self.teams[1].points)
        # print(self.game.cards_on_table)
        # print(self.game.stiche)
        # print(obs)
        # self.player.set_action(action)
        #
        # time.sleep(100)

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
        observation = self.observation_dict_to_index(self.observation)
        episode_over = self.observation['teams'][0]['points'] + self.observation['teams'][1]['points'] == 157
        reward = self._get_reward(episode_over, rules_reward=True)
        return observation, reward, episode_over, {}

    def reset(self):
        """Resets the state of the environment and returns an initial observation.

                Returns: observation (object): the initial observation of the
                    space.
        """
        logger.info("resetting the environment")

        # self.tournament.teams[0].points = 0
        # self.tournament.teams[1].points = 0
        self.observation = {}

        self._control_endless_play()  # if this is not called here, the endless play blocks the execution

        wait = True
        if self.observation == {}:
            wait = False
        observation = self.player.get_observation(wait)
        self.observation = observation

        return self.observation_dict_to_index(observation)

    def _control_endless_play(self, stop=False):
        self.game.endless_play_control.acquire()
        self.game.stop_playing = stop
        self.game.endless_play_control.notify_all()
        self.game.endless_play_control.release()

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
        """
        Closes the environment and cleans up resources
        :return:
        """
        logger.info("closing the environment")

        self._control_endless_play(stop=True)

        # self._join_threads()

    def _join_threads(self):
        main_thread = threading.current_thread()
        for t in threading.enumerate():
            if t is main_thread:
                continue
            logging.debug('joining %s', t.getName())
            t.join()

    def _take_action(self, action):
        action += 1  # action is sampled between 0 and 35 but must be between 1 and 36!
        action = from_index_to_card(action)
        self.action = action
        self.player.set_action(action)
        self.observation = self.player.get_observation()

    def _get_reward(self, episode_over, rules_reward=False):
        """
        Calculates the reward of the current timestep
        :param episode_over:
        :param rules_reward:
        :return:
        """
        if rules_reward:
            return self._rules_reward()
        else:
            return self._stich_reward(episode_over)

    def _stich_reward(self, episode_over):
        """
        Gives as reward the point difference of the two teams at the end of the episode.
        We hope that this reward enables learning useful tactics.
        :param episode_over:
        :return:    0 when the episode (game) is still running
                    the point difference of the game between the team of the RL player and the opponent team
        """
        if episode_over:
            # reward = self.observation['teams'][0]['points'] - self.observation['teams'][1]['points']
            return self.tournament.teams[0].points - self.tournament.teams[1].points
        else:
            return 0

    def _rules_reward(self):
        """
        Gives as reward 1 when the action the RL player chose is allowed and -1 otherwise.
        We hope that with this reward we can make the RL player learn the rules of the game.
        :return:    1 when the action is allowed
                    -1 otherwise
        """
        allowed_cards = self.player.allowed_cards(self.observation)
        if self.action not in allowed_cards:
            return -1
        else:
            return 1

    @staticmethod
    def observation_dict_to_tuple(observation):
        hand = [(4, 9)] * 9
        for i in range(len(observation["cards"])):
            hand[i] = from_card_to_tuple(observation["cards"][i])
        return tuple(hand)

    @staticmethod
    def observation_dict_to_index(observation):
        hand = [0] * 9
        for i in range(len(observation["cards"])):
            hand[i] = from_card_to_index(observation["cards"][i])

        # leave stack for simplicity for now
        stack = [0] * (9 * 4)
        for i in range(len(observation["stiche"])):
            stich = observation["stiche"][i]["played_cards"]
            for j in range(len(stich)):
                stack[4 * i + j] = from_string_to_index(stich[j]['card'])

        table = [0] * 3
        for i in range(len(observation["table"])):
            table[i] = from_string_to_index(observation["table"][i]["card"])

        return hand + stack + table
