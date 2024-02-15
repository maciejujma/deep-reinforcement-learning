import gym
import pybullet_envs
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.env_util import make_vec_env

from src.algorithms.hyperparameters import ppo_stable_baselines_hyperparameters

# hard-coded arguments
TENSORBOARD_LOGS_PATH: str = "logs/tensorboard"
ENV_ID: str = "AntBulletEnv-v0"
NUMBER_OF_ENVIRONMENTS: int = 4
RANDOM_SEED: str = 23
TRAINING_STEPS: int = 200000
PATH_TO_SAVE_MODEL: str = f"data/models/ppo-{ENV_ID}"
PATH_TO_SAVE_ENVIRONMENT: str = f"data/environments/vec_normalize-{ENV_ID}.pkl"

# Parallel environments
environment = make_vec_env(env_id=ENV_ID,
                           n_envs=NUMBER_OF_ENVIRONMENTS,
                           seed=RANDOM_SEED)

# adding this wrapper to normalize the observation and the reward
# environment = VecNormalize(environment,
#                            norm_obs=True,
#                            norm_reward=True,
#                            clip_obs=10.0)


model = PPO(policy="MlpPolicy",
            env=environment,
            verbose=1,
            tensorboard_log=TENSORBOARD_LOGS_PATH,
            batch_size=ppo_stable_baselines_hyperparameters['batch_size'],
            clip_range=ppo_stable_baselines_hyperparameters['clip_range'],
            ent_coef=ppo_stable_baselines_hyperparameters['ent_coef'],
            gae_lambda=ppo_stable_baselines_hyperparameters['gae_lambda'],
            gamma=ppo_stable_baselines_hyperparameters['gamma'],
            learning_rate=ppo_stable_baselines_hyperparameters['learning_rate'],
            max_grad_norm=ppo_stable_baselines_hyperparameters['max_grad_norm']
            )

model.learn(total_timesteps=TRAINING_STEPS)

"""
to see the logs during the training type in the terminal:
>>> tensorboard --logdir logs/tensorboard
"""

# saving the model and environment statistics
# model.save(PATH_TO_SAVE_MODEL)
# environment.save(PATH_TO_SAVE_ENVIRONMENT)

# evaluation
# loading the saved statistics
eval_env = DummyVecEnv([lambda: gym.make(ENV_ID)])
# eval_env = VecNormalize.load(PATH_TO_SAVE_ENVIRONMENT, eval_env)


#  do not update them at test time
eval_env.training = False
# reward normalization is not needed at test time
eval_env.norm_reward = False

# loading the trained model
model = PPO.load(PATH_TO_SAVE_MODEL)

mean_reward, std_reward = evaluate_policy(model, eval_env)

print(f"Mean reward = {mean_reward}. \n Std of the reward: {std_reward}")
