"""
This script shows how to use the RL Baselines3 Zoo framework.
RL Baselines3 Zoo is a collection of pre-trained Reinforcement Learning agents using Stable-Baselines3.
It also provides basic scripts for training, evaluating agents, tuning hyperparameters and recording videos.

Framework's url: https://stable-baselines3.readthedocs.io/en/master/index.html
"""

"""
go to the folder with pre-trained agents

>>> cd rl-baselines3-zoo

we train DQN model using hyper parameters from rl-baselines3-zoo/hyperparams/dqn.yml on SpaceInvader environment
the trained model will be saved in rl-baselines3-zoo/logs folder

>>> python train.py --algo dqn  --env SpaceInvadersNoFrameskip-v4 -f logs/

evaluation of the trained model (5000 steps)

>>> python enjoy.py  --algo dqn  --env SpaceInvadersNoFrameskip-v4  --no-render  --n-timesteps 5000  --folder logs/
"""
