from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import torch.optim


class Policy(nn.Module):
    """
    Architecture of the policy.
    """

    def __init__(self, state_size: int, action_size: int, hidden_layer_size: int) -> None:
        super(Policy, self).__init__()
        self.full_connected_layer_1 = nn.Linear(state_size, hidden_layer_size)
        self.full_connected_layer_2 = nn.Linear(hidden_layer_size, action_size)

    def forward(self, x) -> torch.Tensor:
        x = self.full_connected_layer_1(x)
        x = F.relu(x)
        x = self.full_connected_layer_2(x)
        return F.softmax(x, dim=1)

    def act(self, state: np.array) -> tuple[int, float]:
        # creating tensor from numpy array
        state = torch.from_numpy(state).float().unsqueeze(0)

        # calculating policy (probabilities of each action for given state)
        probabilities = self.forward(state)

        # creating categorical/multinomial distribution
        m = Categorical(probabilities)
        action = m.sample()

        return action.item(), m.log_prob(action)


def reinforce_algorithm(policy: Policy,
                        optimizer: torch.optim,
                        n_training_episodes: int,
                        environment,
                        max_t: int,
                        gamma: float,
                        print_every: int):
    """
    Implementation of the reinforce algorithm, also called monte-carlo policy gradient. Uses an estimated return from
    an entire episode to update the policy parameter theta.

    :param policy: Policy

    :param optimizer:
        Optimizer looking for the parameters minimizing the loss of the policy.
    :param n_training_episodes: int
        Number of the training episodes.
    :param environment:
        Training environment from gym library.
    :param max_t: int
        Maximum number of the steps within one episode.
    :param gamma: float
        Discount factor.
    :param print_every: int
        Number of episodes after which to print the average score.

    :return: List[float]
        List of scores for each episode.
    """
    scores = []

    for i_episode in range(1, n_training_episodes + 1):
        saved_log_probabilities = []
        rewards = []
        state = environment.reset()

        for t in range(max_t):

            # using policy to choose the action
            action, log_probability = policy.act(state)
            saved_log_probabilities.append(log_probability)

            # observing environment response for policy's action
            state, reward, done, _ = environment.step(action)
            rewards.append(reward)
            if done:
                break
        scores.append(sum(rewards))

        # calculating return
        returns = deque(maxlen=max_t)
        n_steps = len(rewards)

        for t in range(n_steps)[::-1]:
            disc_return_t = (returns[0] if len(returns) > 0 else 0)
            returns.appendleft(gamma * disc_return_t + rewards[t])

        # standardization of the returns is employed to make training more stable
        eps = np.finfo(np.float32).eps.item()
        # eps is the smallest representable float, which is
        # added to the standard deviation of the returns to avoid numerical instabilities
        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std(dim=0) + eps)

        policy_loss = []
        for log_prob, disc_return in zip(saved_log_probabilities, returns):
            policy_loss.append(-log_prob * disc_return)
        policy_loss = torch.cat(policy_loss).sum()

        # performing optimization step
        optimizer.zero_grad()
        policy_loss.backward()
        optimizer.step()

        if i_episode % print_every == 0:
            print(f'Episode {i_episode}\tAverage Score: {np.mean(scores)}')

    return scores
