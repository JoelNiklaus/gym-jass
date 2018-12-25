import asyncio
import sys
import time
import logging
from multiprocessing import Process

import gym
import gym_jass

from gym import envs


def test_run_once():
    pass


def test_run_twice():
    pass


def test_run_with_processes():
    def learn():
        logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

        env = gym.make('Schieber-v0')
        time.sleep(1)

        for i_episode in range(2):
            observation = env.reset()
            for t in range(10):
                env.render()
                action = env.action_space.sample()
                print(f"action: {action}")
                observation, reward, done, info = env.step(action)
                print(f"observation: {observation}")
                print(f"reward: {reward}")
                print(f"done: {done}")
                if done:
                    print("Episode finished after {} timesteps".format(t + 1))
                    break

        env.close()

        time.sleep(1)

        env = gym.make('Schieber-v0')

        for i_episode in range(2):
            observation = env.reset()
            for t in range(10):
                env.render()
                action = env.action_space.sample()
                print(action)
                observation, reward, done, info = env.step(action)
                print(f"observation: {observation}")
                print(f"reward: {reward}")
                print(f"done: {done}")
                if done:
                    print("Episode finished after {} timesteps".format(t + 1))
                    break

        env.close()

    process1 = Process(target=learn)
    process2 = Process(target=learn)
    process1.start()
    process2.start()


def test_step():
    pass


def test_reset():
    pass


def test_render():
    pass


def test_seed():
    pass


def test_close():
    pass
