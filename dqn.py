import os
from collections import namedtuple
import random
import numpy as np
import torch

import logger
import agent
import networks
import util


Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward',
                          'non_final'))


class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)

        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def empty(self):
        self.memory.clear()

    def __len__(self):
        return len(self.memory)


class DQNAgent(agent.Agent):
    def __init__(self, ob_size, actions, device, gamma, memory_size,
                 layer_sizes=[], adv_sizes=[], val_sizes=[], double=False,
                 dueling=False):
        super().__init__(ob_size, actions, device)

        self.gamma = gamma
        self.memory_size = memory_size
        self.layer_sizes = layer_sizes
        self.adv_sizes = adv_sizes
        self.val_sizes = val_sizes
        self.double = double
        self.dueling = dueling

        if dueling:
            self.policy_net = networks.DuelingNetwork(
                self.ob_size, layer_sizes, adv_sizes, val_sizes,
                self.actions).to(self.device)
            self.target_net = networks.DuelingNetwork(
                self.ob_size, layer_sizes, adv_sizes, val_sizes,
                self.actions).to(self.device)
        else:
            self.policy_net = networks.DenseNetwork(
                self.ob_size, layer_sizes, self.actions).to(self.device)
            self.target_net = networks.DenseNetwork(
                self.ob_size, layer_sizes, self.actions).to(self.device)

        self._target_update()
        self.policy_net.eval()
        self.target_net.eval()

        self.memory = ReplayMemory(memory_size)

    @staticmethod
    def load(model_dir):
        model_params = util.load_pickle(
            os.path.join(model_dir, 'params.pickle'))

        model = DQNAgent(*model_params)

        model.policy_net.load_state_dict(
            torch.load(os.path.join(model_dir, 'policy_net.pt')))
        model._target_update()

        model.memory = util.load_pickle(
            os.path.join(model_dir, 'memory.pickle'))

        return model

    def save(self, save_dir):
        model_params = (self.ob_size, self.actions, self.device,
                        self.gamma, self.memory_size, self.layer_sizes,
                        self.adv_sizes, self.val_sizes, self.double,
                        self.dueling)
        util.save_pickle(os.path.join(save_dir, 'params.pickle'), model_params)

        torch.save(self.policy_net.state_dict(),
                   os.path.join(save_dir, 'policy_net.pt'))

        util.save_pickle(os.path.join(save_dir, 'memory.pickle'), self.memory)

        logger.info(f'Model saved to {save_dir}')


    def _target_update(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def preprocess_input(self, x):
        return torch.from_numpy(x.astype(np.float32)).to(self.device)

    def move(self, state, epsilon=0):
        if np.random.rand() < epsilon:
            return random.randrange(self.actions)

        with torch.no_grad():
            action = torch.argmax(self.policy_net(state)).item()

        return action

    def _optimize_step(self, batch_size, optimizer):
        if len(self.memory) < batch_size:
            return

        transitions = self.memory.sample(batch_size)
        batch = Transition(*zip(*transitions))

        state_batch = torch.vstack(batch.state)
        action_batch = torch.from_numpy(
            np.array(batch.action, dtype=np.int64)).to(self.device).unsqueeze(1)
        reward_batch = torch.from_numpy(
            np.array(batch.reward, dtype=np.float32)).to(self.device)

        non_final_batch = torch.from_numpy(
            np.array(batch.non_final, dtype=np.bool)).to(self.device)
        non_final_next_states = torch.vstack(batch.next_state)[non_final_batch]

        q_values = self.policy_net(state_batch).gather(1, action_batch)

        with torch.no_grad():
            q_target = torch.zeros(batch_size, device=self.device)

            if self.double:
                a_next = torch.argmax(
                    self.policy_net(non_final_next_states).detach(), dim=1)

                q_target[non_final_batch] = self.target_net(
                    non_final_next_states
                ).detach().gather(1, a_next.unsqueeze(1)).squeeze(1)
            else:
                q_target[non_final_batch] = self.target_net(
                    non_final_next_states).detach().max(1)[0]

            q_expected = reward_batch + self.gamma * q_target

        loss = torch.nn.functional.mse_loss(
            q_values, q_expected.unsqueeze(1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    def train(self, env, episodes, episode_maxiter, epsilon, epsilon_final,
              batch_size, target_update, warmup_episodes=None, verbose_step=0):
        self.policy_net.train()

        rewards = []
        avg_rewards = []

        if warmup_episodes is None:
            warmup_episodes = episodes

        eps_drop = (epsilon - epsilon_final) / warmup_episodes

        optimizer = torch.optim.Adam(self.policy_net.parameters())

        steps = 0

        for ep in range(1, episodes + 1):
            state = env.reset()
            state = self.preprocess_input(state)

            reward = 0.

            for _ in range(1, episode_maxiter + 1):
                action = self.move(state, epsilon)
                s_next, r, done, _ = env.step(action)
                s_next = self.preprocess_input(s_next)

                self.memory.push(state, action, s_next, r, not done)

                state = s_next
                reward += r

                self._optimize_step(batch_size, optimizer)

                steps += 1

                if steps % target_update == 0:
                    self._target_update()

                if done:
                    break

            rewards.append(reward)
            avg_rewards.append(np.mean(rewards[-50:]))

            if epsilon > epsilon_final:
                epsilon = max(epsilon - eps_drop, epsilon_final)

            if verbose_step > 0 and ep % verbose_step == 0:
                print(f'[{ep}/{episodes}] steps = {steps}, reward = {reward}, '
                      f'epsilon = {epsilon:.4f}')

            util.plot_values('Rewards',
                             [('reward', rewards), ('avg_reward', avg_rewards)],
                             xlabel='episode', ylabel='reward',
                             legend_location='upper left')

        self.policy_net.eval()
        util.show_plots()
