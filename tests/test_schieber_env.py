import sys
import time
import logging
from multiprocessing import Process
import gym
import gym_jass

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)


def test_run_once():
    run_one_episode()


def test_run_twice():
    run_one_episode()
    time.sleep(1)
    run_one_episode()


def test_run_with_processes():
    process1 = Process(target=run_one_episode)
    process2 = Process(target=run_one_episode)
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


def run_one_episode():
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
