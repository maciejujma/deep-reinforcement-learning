import gym
import pybullet_envs
from stable_baselines3 import A2C
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.env_util import make_vec_env

from src.algorithms.hyperparameters import a2c_stable_baselines_hyperparameters

# hard-coded arguments
TENSORBOARD_LOGS_PATH: str = "../../logs/tensorboard"
ENV_ID: str = "AntBulletEnv-v0"
TRAINING_STEPS: int = 200000
PATH_TO_SAVE_MODEL: str = "../../data/models/a2c-AntBulletEnv-v0"
PATH_TO_SAVE_ENVIRONMENT: str = "../../data/environments/vec_normalize-AntBulletEnv-v0.pkl"

# creating the environment (vector of the environments)
environment = make_vec_env(ENV_ID, n_envs=4)

# adding this wrapper to normalize the observation and the reward
environment = VecNormalize(environment, norm_obs=True, norm_reward=True, clip_obs=10.0)

# creating Advantage Actor-Critic model
model = A2C(policy=a2c_stable_baselines_hyperparameters['policy'],
            env=environment,
            gae_lambda=a2c_stable_baselines_hyperparameters['gae_lambda'],
            gamma=a2c_stable_baselines_hyperparameters['gamma'],
            learning_rate=a2c_stable_baselines_hyperparameters['learning_rate'],
            max_grad_norm=a2c_stable_baselines_hyperparameters['max_grad_norm'],
            n_steps=a2c_stable_baselines_hyperparameters['n_steps'],
            vf_coef=a2c_stable_baselines_hyperparameters['vf_coef'],
            ent_coef=a2c_stable_baselines_hyperparameters['ent_coef'],
            tensorboard_log=TENSORBOARD_LOGS_PATH,
            policy_kwargs=dict(
                log_std_init=a2c_stable_baselines_hyperparameters['policy_kwargs']['log_std_init'],
                ortho_init=a2c_stable_baselines_hyperparameters['policy_kwargs']['ortho_init']
            ),
            normalize_advantage=a2c_stable_baselines_hyperparameters['normalize_advantage'],
            use_rms_prop=a2c_stable_baselines_hyperparameters['use_rms_prop'],
            use_sde=a2c_stable_baselines_hyperparameters['use_sde'],
            verbose=a2c_stable_baselines_hyperparameters['verbose']
            )

# training the model
model.learn(TRAINING_STEPS)

"""
to see the logs during the training type in the terminal:
>>> tensorboard --logdir logs/tensorboard
"""

# saving the model and environment statistics
model.save(PATH_TO_SAVE_MODEL)
environment.save(PATH_TO_SAVE_ENVIRONMENT)

# evaluation
# loading the saved statistics
eval_env = DummyVecEnv([lambda: gym.make(ENV_ID)])
eval_env = VecNormalize.load(PATH_TO_SAVE_ENVIRONMENT, eval_env)

#  do not update them at test time
eval_env.training = False
# reward normalization is not needed at test time
eval_env.norm_reward = False

# loading the trained model
model = A2C.load(PATH_TO_SAVE_MODEL)

mean_reward, std_reward = evaluate_policy(model, eval_env)

print(f"Mean reward = {mean_reward}. \n Std of the reward: {std_reward}")
