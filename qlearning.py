import random
import numpy as np

import agent
import util


class QLearningAgent(agent.Agent):
    def __init__(self, ob_size, actions, device, gamma):
        super().__init__(ob_size, actions, device)

        self.gamma = gamma
        self.Q = {}

    def move(self, state, epsilon=0):
        if np.random.rand() < epsilon or not self.Q:
            return random.randrange(self.actions)

        return np.random.choice(self._get_best_actions(state))

    def _get_best_actions(self, state):
        best_q = None
        best_actions = None

        for a in range(self.actions):
            current_q = self.Q.get((state, a), 0.)

            if best_q is None or current_q > best_q:
                best_q = current_q
                best_actions = [a]
            elif abs(current_q - best_q) < 1e-9:
                best_actions.append(a)

        return best_actions

    def _get_max_q(self, state):
        max_q = None

        for a in range(self.actions):
            current_q = self.Q.get((state, a), 0.)

            if max_q is None or current_q > max_q:
                max_q = current_q

        return max_q

    def _update_Q(self, s, a, r, s_next, done, alpha):
        self.Q[s, a] = (1. - alpha) * self.Q.get((s, a), 0.) + alpha * r

        if not done:
            self.Q[s, a] += alpha * self.gamma * self._get_max_q(s_next)

    def train(self, env, episodes, episode_maxiter, alpha, alpha_decay, epsilon,
              epsilon_final, warmup_episodes=None, verbose_step=0):
        rewards = []
        avg_rewards = []

        if warmup_episodes is None:
            warmup_episodes = episodes

        eps_drop = (epsilon - epsilon_final) / warmup_episodes

        steps = 0

        for ep in range(1, episodes + 1):
            state = env.reset()
            reward = 0.

            for _ in range(1, episode_maxiter + 1):
                action = self.move(state, epsilon)
                s_next, r, done, _ = env.step(action)

                self._update_Q(state, action, r, s_next, done, alpha)

                state = s_next
                reward += r

                steps += 1

                if done:
                    break

            rewards.append(reward)
            avg_rewards.append(np.mean(rewards[-50:]))

            alpha *= alpha_decay
            if epsilon > epsilon_final:
                epsilon = max(epsilon - eps_drop, epsilon_final)

            if verbose_step > 0 and ep % verbose_step == 0:
                print(f'[{ep}/{episodes}] steps = {steps}, reward = {reward}, '
                      f'alpha = {alpha:.4f}, epsilon = {epsilon:.4f}, '
                      f'Qsize = {len(self.Q)}')

            util.plot_values('Rewards',
                             [('reward', rewards), ('avg_reward', avg_rewards)],
                             xlabel='episode', ylabel='reward',
                             legend_location='upper left')
