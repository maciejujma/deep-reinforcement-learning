import numpy as np


def agent_evaluation(evaluation_environment, max_steps, n_evaluation_episodes, policy):
    """
    Policy based agents evaluating function.

    :param evaluation_environment:
        Evaluation environment from gym library.
    :param max_steps: int
        Number of maximum steps in a single episode.
    :param n_evaluation_episodes: int
        Number of evaluation episodes.
    :param policy: torch.nn.Module
        Policy to evaluate.
    :return: Tuple[float]
        Mean and standard deviation of the all rewards.
    """

    all_episodes_rewards = []

    for episode in range(n_evaluation_episodes):
        episode_rewards = []
        state = evaluation_environment.reset()

        for step in range(max_steps):

            action, _ = policy.act(state)
            state, reward, done, _ = evaluation_environment.step(action)

            episode_rewards.append(reward)
            if done:
                break
        all_episodes_rewards.append(sum(episode_rewards))

    return np.mean(all_episodes_rewards), np.std(all_episodes_rewards)
