import numpy as np
import gym


class DiscretizedObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env, n_bins, low=None, high=None):
        super(DiscretizedObservationWrapper, self).__init__(env)

        low = np.array(self.observation_space.low if low is None else low)
        high = np.array(self.observation_space.high if high is None else high)

        self.n_bins = n_bins
        self.val_bins = [np.linspace(l, h, n_bins + 1) for l, h in
                         zip(low.flatten(), high.flatten())]
        self.observation_space = gym.spaces.Discrete(n_bins**len(low))

    def _convert_to_one_number(self, digits):
        return sum([d * ((self.n_bins + 1) ** i) for i, d in enumerate(digits)])

    def observation(self, observation):
        digits = [np.digitize([x], bins)[0]
                  for x, bins in zip(observation.flatten(), self.val_bins)]

        return self._convert_to_one_number(digits)


class FlatActionWrapper(gym.ActionWrapper):
    def __init__(self, env):
        super(FlatActionWrapper, self).__init__(env)

        self.ns = []
        self.n = 1

        for a in env.action_space:
            self.n *= a.n
            self.ns.append(a.n)

        self.action_space = gym.spaces.Discrete(self.n)

    def _num_to_tuple(self, num):
        tup = []

        for n in self.ns:
            tup.append(num % n)
            num //= n

        return tuple(tup)

    def action(self, action):
        return self._num_to_tuple(action)