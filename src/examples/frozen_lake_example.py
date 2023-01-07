from src.algorithms.hyperparameters import q_learning_hyperparameters
from src.algorithms.q_learning import *
from src.evalutation import evaluate_q_learning_agent

"""
    Example of solving the FrozenLake-v1 environment using Q-Learning
    algorithm.
"""


# creating environment
frozen_lake_environment = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=False)
state_space_size = frozen_lake_environment.observation_space.n
action_space_size = frozen_lake_environment.action_space.n

# creating algorithm
q_learning_algorithm = QLeaningAlgorithm(environment=frozen_lake_environment)

# training model
trained_q_table = q_learning_algorithm.train(n_training_episodes=q_learning_hyperparameters['n_training_episodes'],
                                             max_steps=q_learning_hyperparameters['max_t'],
                                             learning_rate=q_learning_hyperparameters['learning_rate'],
                                             discount_factor=q_learning_hyperparameters['gamma'],
                                             decay_rate=q_learning_hyperparameters['decay_rate'],
                                             max_epsilon=q_learning_hyperparameters['max_epsilon'],
                                             min_epsilon=q_learning_hyperparameters['min_epsilon'])

# evaluation q_table - zeros as q-values
evaluation_q_table = np.zeros((state_space_size, action_space_size))

mean_reward, std_reward = evaluate_q_learning_agent(env=frozen_lake_environment,
                                                    max_steps=q_learning_hyperparameters['n_evaluation_episodes'],
                                                    n_eval_episodes=q_learning_hyperparameters['max_t'],
                                                    Q=trained_q_table,
                                                    seed=[])

eval_mean_reward, eval_std_reward = \
    evaluate_q_learning_agent(env=frozen_lake_environment,
                              max_steps=q_learning_hyperparameters['n_evaluation_episodes'],
                              n_eval_episodes=q_learning_hyperparameters['max_t'],
                              Q=evaluation_q_table,
                              seed=[])

print(f"\nTrained algorithm: Mean: {mean_reward}, Std: {std_reward}")
print(f"Not trained algorithm: Mean: {eval_mean_reward}, Std: {eval_std_reward}")
