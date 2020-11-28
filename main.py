#!/usr/bin/env python3

import gym
import numpy as np


def main():
    env = gym.make('Copy-v0')
    env.reset()

    print(f'Observation space:\n{env.observation_space}\n\n'
          f'Action space:\n{env.action_space}')

    if False:
        return

    t = 0
    done = False
    while not done:
        env.render()

        _, _, done, _ = env.step(env.action_space.sample())
        t += 1

    print(f'Episode finished after {t} timesteps.')

    env.close()


if __name__ == '__main__':
    main()
