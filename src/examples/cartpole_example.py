import gym

from src.algorithms.hyperparameters import *
from src.algorithms.reinforce import *
from src.evalutation import agent_evaluation

"""
    Example of solving the CartPole-v1 environment using policy gradient
    algorithm REINFORCE (a.k.a. monte-carlo policy gradient)
"""

# creating environment
ENVIRONMENT_ID = "CartPole-v1"
cartpole_environment = gym.make(ENVIRONMENT_ID)
cartpole_evaluation_environment = gym.make(ENVIRONMENT_ID)
state_space_size = cartpole_environment.observation_space.shape[0]
action_space_size = cartpole_environment.action_space.n

# defining the policy and the optimizer
cartpole_policy = Policy(state_size=state_space_size,
                         action_size=action_space_size,
                         hidden_layer_size=cartpole_hyperparameters['h_size'])
cartpole_optimizer = torch.optim.Adam(params=cartpole_policy.parameters(),
                                      lr=cartpole_hyperparameters['learning_rate'])

# training the algorithm
reinforce_algorithm(policy=cartpole_policy,
                    optimizer=cartpole_optimizer,
                    n_training_episodes=cartpole_hyperparameters["n_training_episodes"],
                    environment=cartpole_environment,
                    max_t=cartpole_hyperparameters["max_t"],
                    gamma=cartpole_hyperparameters["gamma"],
                    print_every=100)

# evaluating the algorithm
trained_policy_evaluation = agent_evaluation(evaluation_environment=cartpole_evaluation_environment,
                                             max_steps=cartpole_hyperparameters["max_t"],
                                             n_evaluation_episodes=cartpole_hyperparameters["n_evaluation_episodes"],
                                             policy=cartpole_policy)

blank_policy_evaluation = agent_evaluation(evaluation_environment=cartpole_evaluation_environment,
                                           max_steps=cartpole_hyperparameters["max_t"],
                                           n_evaluation_episodes=cartpole_hyperparameters["n_evaluation_episodes"],
                                           policy=Policy(state_size=state_space_size,
                                                         action_size=action_space_size,
                                                         hidden_layer_size=cartpole_hyperparameters['h_size']))

print(f"\nTrained algorithm: Mean: {trained_policy_evaluation[0]}, Std: {trained_policy_evaluation[1]}")
print(f"Not trained algorithm: Mean: {blank_policy_evaluation[0]}, Std: {blank_policy_evaluation[1]}")
