from tqdm import tqdm

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


def evaluate_q_learning_agent(env, max_steps, n_eval_episodes, Q, seed):
    episode_rewards = []
    for episode in tqdm(range(n_eval_episodes)):
        if seed:
            state = env.reset(seed=seed[episode])
        else:
            state = env.reset()
        total_rewards_ep = 0

        for step in range(max_steps):
            # Take the action (index) that have the maximum expected future reward given that state
            action = np.argmax(Q[state][:])
            new_state, reward, done, info = env.step(action)
            total_rewards_ep += reward

            if done:
                break
            state = new_state
        episode_rewards.append(total_rewards_ep)
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)

    return mean_reward, std_reward
