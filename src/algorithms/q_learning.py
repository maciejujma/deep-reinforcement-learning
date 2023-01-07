from tqdm import tqdm

import gym
import numpy as np


class QLeaningAlgorithm:
    q_table: np.array
    environment: gym.Env

    def __init__(self, environment):
        self.environment = environment
        self.q_table = self._init_q_table(state_space_size=self.environment.observation_space.n,
                                          action_space_size=self.environment.action_space.n)

    @staticmethod
    def _init_q_table(state_space_size: int, action_space_size: int):
        q_table = np.ones((state_space_size, action_space_size)) * 0.2  # "optimistic initial values"
        return q_table

    def _greedy_policy(self, state):
        action = np.argmax(self.q_table[state])
        return action

    def _epsilon_greedy_policy(self, state, epsilon):

        if epsilon < np.random.uniform(0, 1):
            action = self._greedy_policy(state=state)
        else:
            action = self.environment.action_space.sample()

        return action

    def _update_q_values(self,
                         state: np.array,
                         new_state: np.array,
                         action,
                         reward,
                         discount_factor,
                         learning_rate):

        # Update Q(s,a):= Q(s,a) + lr [R(s,a) + gamma * max Q(s',a') - Q(s,a)]
        temporal_difference_target = reward + discount_factor * self.q_table[new_state][self._greedy_policy(new_state)]
        temporal_difference = temporal_difference_target - self.q_table[state][action]

        self.q_table[state][action] = self.q_table[state][action] + learning_rate * temporal_difference

    def train(self, n_training_episodes: int,
              max_steps: int,
              learning_rate: float,
              discount_factor: float,
              decay_rate: float,
              min_epsilon: float,
              max_epsilon: float):

        for episode in tqdm(range(n_training_episodes)):

            # epsilon at the beginning should be bigger to encourage policy to explore (exploration > exploitation)
            # later policy should more exploit the best actions (exploitation > exploration)
            epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate * episode)
            state = self.environment.reset()

            for step in range(max_steps):

                # using policy to choose the action
                action = self._epsilon_greedy_policy(state, epsilon)

                # observing environment response for policy's action
                new_state, reward, done, _ = self.environment.step(action)

                # updating q-values using observed rewards
                self._update_q_values(state, new_state, action, reward, discount_factor, learning_rate)

                # if reached termination state break the episode
                if done:
                    break
                state = new_state
        return self.q_table
