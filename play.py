#!/usr/bin/env python3

import gym

import config
import dqn


def main(args):
    agent = dqn.DQNAgent.load(args.model_dir)

    env = gym.make('CartPole-v1')

    for i_episode in range(20):
        observation = env.reset()
        done = False
        steps = 0

        while not done:
            env.render()

            observation = agent.preprocess_input(observation)
            action = agent.move(observation)

            observation, reward, done, info = env.step(action)

            steps += 1

        print(f'Episode finished after {steps} timesteps')

    env.close()


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser('./main.py')
    
    parser.add_argument('-m', '--model_dir', type=str, required=True,
                        help='directory where the agent is saved')
    
    args = parser.parse_args()

    main(args)
