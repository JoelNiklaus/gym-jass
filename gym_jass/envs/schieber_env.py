import logging
import threading

import gym
import numpy as np
from gym import spaces
from gym.spaces import Discrete
from schieber.card import from_card_to_tuple, from_card_to_index, from_string_to_index, from_card_to_onehot, \
    from_string_to_onehot
from schieber.game import Game
from schieber.player.challenge_player.challenge_player import ChallengePlayer
from schieber.player.greedy_player.greedy_player import GreedyPlayer
from schieber.player.random_player import RandomPlayer
from schieber.player.external_player import ExternalPlayer
from schieber.team import Team

logger = logging.getLogger(__name__)


# TODO start thread with eternal game only in first step in order to enable seeding of the environment


class CardSpace(gym.spaces.MultiBinary):
    def __init__(self):
        self.n = 1

    def sample(self):
        # trump is intentionally left away for now because the it could sample different trumps in the same game
        n_values = 9
        n_suits = 4
        suit = np.eye(n_suits)[gym.spaces.np_random.choice(n_suits, 1)]
        value = np.eye(n_values)[gym.spaces.np_random.choice(n_values, 1)]
        return np.append(suit, value)


class SchieberEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    # our points minus opponent team's points
    # reward is always given at the end of one game
    # no match bonus is awarded to make it a constant-sum game
    reward_range = (-157, 157)

    # 0 stands for NO CARD, 1 to 36 are mapped to the 36 possible cards
    # the one card to be chosen
    # action_space = spaces.Discrete(36)
    # action_space = CardSpace()
    action_space = Discrete(9)  # index of the card on the hand

    # index 0 to 8: the player's hand
    # index 9 to 44: the cards in the stack which have been turned over in the order of appearance
    # index 45 to 47: the cards currently on the table: 45 --> 1st card, 46 --> 2nd card, 47 --> 3rd card
    # observation_space = spaces.Box(low=0, high=36, shape=(48,), dtype=int)  # card encoded as scalar between 0 and 36

    # This space does not need to be extended by an own implementation because it is normally not sampled from the observation space.
    observation_space = spaces.Box(low=0, high=1, shape=(48, 13,), dtype=int)  # card encoded as two hot vector

    def __init__(self, reward_function='play', trumps='all', partner_level='greedy', opponents_level='greedy'):
        """
        Initialize the environment. Starts an endless game.
        :param reward_function: 'play' for good play, 'rules' for playing valid cards
        :param trumps: 'all' for all available trumps, 'obe_abe' for only obe_abe
        :param partner_level: 'greedy' or 'random'
        :param opponents_level: 'greedy' or 'random'
        """
        super(SchieberEnv, self).__init__()

        self.reward_function = reward_function
        self.trumps = trumps

        self.action = None
        self.observation = {}

        self.reward = 0
        self.episode_over = False
        self.valid_card_played = None

        self.player = ExternalPlayer(name='GYM-RL', trumps=trumps)

        teams = self.init_teams(opponents_level, partner_level, trumps)

        self.game = Game(teams, point_limit=1000, use_counting_factor=False, seed=1)

        thread = threading.Thread(target=self.game.play_endless)
        thread.daemon = True  # make thread a daemon so it is killed when the main thread exits
        thread.start()

    def init_teams(self, opponents_level, partner_level, trumps):
        global partner, opponent1, opponent2

        if partner_level == 'greedy':
            partner = GreedyPlayer(name='GreedyPlayer', trumps=trumps)
        elif partner_level == 'random':
            partner = RandomPlayer(name='RandomPartner', trumps=trumps)
        else:
            logger.error("Please specify a valid partner_level")

        if opponents_level == 'greedy':
            opponent1 = GreedyPlayer(name='GreedyOpponent1', trumps=trumps)
            opponent2 = GreedyPlayer(name='GreedyOpponent2', trumps=trumps)
        elif opponents_level == 'random':
            opponent1 = RandomPlayer(name='RandomOpponent1', trumps=trumps)
            opponent2 = RandomPlayer(name='RandomOpponent2', trumps=trumps)
        else:
            logger.error("Please specify a valid opponents_level")

        team_1 = Team(players=[self.player, partner])
        team_2 = Team(players=[opponent1, opponent2])
        return [team_1, team_2]

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
        self.episode_over = not self.observation['cards']  # this is true when the list is empty
        self.reward = self._get_reward()
        logger.info(self.render())  # make rendering available during training too
        return self.observation_dict_to_onehot_matrix(self.observation), self.reward, self.episode_over, {}

    def reset(self):
        """Resets the state of the environment and returns an initial observation.

                Returns: observation (object): the initial observation of the
                    space.
        """
        logger.info("resetting the environment")

        self.observation = {}

        self._control_endless_play(stop=False)  # if this is not called here, the endless play blocks the execution

        observation = self.player.get_observation()
        self.observation = observation

        return self.observation_dict_to_onehot_matrix(observation)

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

        stiche = self.observation['stiche']
        stich = "No Stich available"
        if stiche:
            stich = stiche[-1]
        output = f"Reward: {self.reward}, " \
            f"Chosen Card: {self.action} --> Card Allowed: {self.valid_card_played}, " \
            f"Played Stich: {stich}"
        print(output)
        return output

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

    def _take_action(self, action):
        """
        Performs the necessary steps to take an action in the environment
            - convert the action intex to a card object
            - set the action in the external player
            - receive the observation from the external player
        :param action:
        :return:
        """
        # self.action = from_index_to_card(action + 1)  # action is sampled between 0 and 35 but must be between 1 and 36!
        # self.action = from_onehot_to_card(action)  # action is a one-hot encoded card
        hand_cards = self.observation["cards"]
        if len(hand_cards) < 1:
            logger.error("The round is over. There are no cards left to play.")
        elif len(hand_cards) == 1:
            logger.error("There is only one card left to play. There is no choice anymore.")
            self.action = hand_cards[0]
        else:
            logger.info(f"There are {len(hand_cards)} cards on the hand.")
            if action < 0:
                logger.error(f"The chosen action index ({action}) is not allowed to be negative!")
            elif action >= len(hand_cards):
                logger.error(f"The chosen action index ({action}) has to be smaller than the number of cards "
                             f"on the hand ({len(hand_cards)})! Choosing the first card on the hand now!")
                self.action = hand_cards[0]
            else:
                logger.info("Successfully chose a card!")
                self.action = hand_cards[action]
        self.valid_card_played = self._is_card_allowed()  # call here, because the new observation is not available yet!
        self.player.set_action(self.action)
        self.observation = self.player.get_observation()

    def _get_reward(self):
        """
        Calculates the reward of the current timestep
        :return:
        """
        if self.reward_function == 'play':
            return self._play_reward()
        elif self.reward_function == 'rules':
            return self._rules_reward()
        else:
            logger.error("Please specify a valid reward function. Choosing play reward now!")
            return self._play_reward()

    def _play_reward(self):
        """
        Gives as reward the point difference of the two teams at the end of the episode.
        We hope that this reward enables learning useful tactics.
        :return:    0 when the episode (game) is still running
                    the point difference of the game between the team of the RL player and the opponent team
        """
        if self.episode_over:
            # reward = self.observation['teams'][0]['points'] - self.observation['teams'][1]['points']
            # return self.tournament.teams[0].points - self.tournament.teams[1].points
            return self.game.teams[0].points - self.game.teams[1].points
        else:
            return 0

    def _rules_reward(self):
        """
        Gives as reward 1 when the action the RL player chose is allowed and -1 otherwise.
        We hope that with this reward we can make the RL player learn the rules of the game.
        :return:    1 when the action is allowed
                    -1 otherwise
        """
        if self.valid_card_played:
            return 1
        else:
            return -1

    def _is_card_allowed(self):
        """
        Checks if the card is allowed. Attention: Take care when to call this method!
        When the new observation is already available, a call to this method will result in a wrong output!
        :return:
        """
        return self.action in self.player.allowed_cards(self.observation)

    @staticmethod
    def observation_dict_to_onehot_matrix(observation):
        hand = SchieberEnv.create_empty_list_of_cards(9)
        if "cards" in observation.keys():  # in the initial observation this may still be empty
            for i in range(len(observation["cards"])):
                hand[i] = from_card_to_onehot(observation["cards"][i])

        stack = SchieberEnv.create_empty_list_of_cards(9 * 4)
        for i in range(len(observation["stiche"])):
            stich = observation["stiche"][i]["played_cards"]
            for j in range(len(stich)):
                stack[4 * i + j] = from_string_to_onehot(stich[j]['card'])

        table = SchieberEnv.create_empty_list_of_cards(3)
        for i in range(len(observation["table"])):
            table[i] = from_string_to_onehot(observation["table"][i]["card"])

        return hand + stack + table

    @staticmethod
    def create_empty_list_of_cards(n_cards):
        card = [0] * 4 + [0] * 9  # trumpf bit not included yet because in a first stage we only care about obe_abe
        return [card for _ in range(n_cards)]

    @staticmethod
    def observation_dict_to_tuple(observation):
        """
        Not used at the moment due to a change in the observation space.
        :param observation:
        :return:
        """
        hand = [(4, 9)] * 9
        for i in range(len(observation["cards"])):
            hand[i] = from_card_to_tuple(observation["cards"][i])
        return tuple(hand)

    @staticmethod
    def observation_dict_to_index(observation):
        """
        Not used at the moment due to a change in the observation space.
        :param observation:
        :return:
        """
        hand = [0] * 9
        if "cards" in observation.keys():  # in the initial observation this may still be empty
            for i in range(len(observation["cards"])):
                hand[i] = from_card_to_index(observation["cards"][i])

        stack = [0] * (9 * 4)
        for i in range(len(observation["stiche"])):
            stich = observation["stiche"][i]["played_cards"]
            for j in range(len(stich)):
                stack[4 * i + j] = from_string_to_index(stich[j]['card'])

        table = [0] * 3
        for i in range(len(observation["table"])):
            table[i] = from_string_to_index(observation["table"][i]["card"])

        return hand + stack + table
