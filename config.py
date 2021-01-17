import json
import gym
import torch

import logger
import wrappers


def get_agent(name, *args, **kwargs):
    if name == 'QLearningAgent':
        import qlearning
        return qlearning.QLearningAgent(*args, **kwargs)
    elif name == 'DQNAgent':
        import dqn
        return dqn.DQNAgent(*args, **kwargs)
    else:
        raise Exception('Invalid agent name.')


def get_wrapper(name, *args, **kwargs):
    if name == 'DiscretizedObservationWrapper':
        return wrappers.DiscretizedObservationWrapper(*args, **kwargs)
    elif name == 'FlatActionWrapper':
        return wrappers.FlatActionWrapper(*args, **kwargs)
    else:
        raise Exception('Invalid wrapper name.')


def apply_wrappers(env, wrapper_list):
    for name, params in wrapper_list:
        env = get_wrapper(name, env, **params)
        logger.info(f'Applied wrapper: {name}')

    return env


def get_observation_size(observation_space):
    if isinstance(observation_space, gym.spaces.Box):
        return observation_space.shape[0]
    elif isinstance(observation_space, gym.spaces.Discrete):
        return observation_space.n


def load_config(file_path):
    def config(env_name, agent_name, agent_params=None,
               train_params=None, wrapper_list=[]):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f'Using device: {device}')
        
        threads = 2
        torch.set_num_threads(threads)
        logger.info(f'Running on {threads} threads')

        env = gym.make(env_name)
        logger.info(f'Loaded environment: {env_name}')

        env = apply_wrappers(env, wrapper_list)
        
        logger.info(f'Observation space: {env.observation_space}')
        logger.info(f'Action space: {env.action_space}')
        
        ob_size = get_observation_size(env.observation_space)
        n_actions = env.action_space.n

        agent = get_agent(agent_name, ob_size, n_actions, **agent_params,
                          device=device)

        logger.info(f'Loaded agent: {agent_name}')
        
        return device, env, agent, train_params

    with open(file_path, 'r') as f:
        cfg = json.load(f)

    return config(**cfg)
