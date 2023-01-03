import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


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

    def act(self, state: np.array) -> tuple[int, float] :
        # creating tensor from numpy array
        state = torch.from_numpy(state).float().unsqueeze(0)

        # calculating policy (probabilities of each action for given state)
        probabilities = self.forward(state)

        # creating categorical/multinomial distribution
        m = Categorical(probabilities)
        action = m.sample()

        return action.item(), m.log_prob(action).item()


torch.manual_seed(23)
np.random.seed(23)
policy = Policy(state_size=4, action_size=2, hidden_layer_size=64)
print(policy.act(np.random.uniform(-1, 1, 4)))


def reinforce_algorithm():
    """
    Implementation of the reinforce algorithm, also called monte-carlo policy gradient. Uses an estimated return from
    an entire episode to update the policy parameter theta.

    :return:
    """
