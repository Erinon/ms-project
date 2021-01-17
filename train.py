#!/usr/bin/env python3

import config


def main(args):
    _, env, agent, train_params = config.load_config(args.config_path)

    agent.train(env, **train_params, verbose_step=10)

    if args.save_dir is not None:
        agent.save(args.save_dir)

    env.close()


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser('./main.py')
    
    parser.add_argument('-c', '--config_path', type=str, required=True,
                        help='path to the configuration file')
    parser.add_argument('-s', '--save_dir', type=str, required=False,
                        help='directory where to save the agent')
    
    args = parser.parse_args()

    main(args)
