import asyncio
import gc
import sys
import time

import gym
import gym_jass

from gym import envs
print(envs.registry.all())
import logging

logging.basicConfig(stream=sys.stdout, level=logging.WARNING)

env = gym.make('Schieber-v0')
time.sleep(1)

for i_episode in range(5):
    observation = env.reset()
    for t in range(100):
        env.render()
        action = env.action_space.sample()
        print(f"action: {action}")
        observation, reward, done, info = env.step(action)
        print(f"observation: {observation}")
        print(f"reward: {reward}")
        print(f"done: {done}")
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break

env.close()

time.sleep(1)

env = gym.make('Schieber-v0')

for i_episode in range(5):
    observation = env.reset()
    for t in range(100):
        env.render()
        action = env.action_space.sample()
        print(action)
        observation, reward, done, info = env.step(action)
        print(f"observation: {observation}")
        print(f"reward: {reward}")
        print(f"done: {done}")
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break

env.close()

for task in asyncio.Task.all_tasks():
    print(task)