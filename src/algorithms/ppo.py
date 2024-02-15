import argparse
import random
import time

import numpy as np
import gym
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name',
                        type=str,
                        default='exp_name',
                        help='name of the experiment')
    parser.add_argument("--gym-id",
                        type=str,
                        default="CartPole-v1",
                        help="the id of the gym environment")
    parser.add_argument("--learning-rate",
                        type=float,
                        default=2.5e-4,
                        help="the learning rate of the optimizer")
    parser.add_argument("--seed",
                        type=int,
                        default=23,
                        help="seed of the experiment")
    parser.add_argument("--total-timesteps",
                        type=int,
                        default=25000,
                        help="total timesteps of the experiments")
    parser.add_argument("--torch_deterministic",
                        type=bool,
                        default=True,
                        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--cuda",
                        type=bool,
                        default=False,
                        help="if changed, cude will be enabled")
    parser.add_argument("--capture_video",
                        type=bool,
                        default=False,
                        help="if changed the video will be saved")

    # algorithm specific parameters
    parser.add_argument("--num-envs",
                        type=int,
                        default=4,
                        help="the number of parallel game environments")
    parser.add_argument("--num-steps",
                        type=int,
                        default=128,
                        help="the number of steps to run in each environment per policy rollout")
    args = parser.parse_args()
    args.batch_size = int(args.num_envs * args.num_steps)
    return args


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, envs) -> None:
        super(Agent, self).__init__()

        # critict network
        self.critic = nn.Sequential(
            layer_init(
                nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )

        # actor network
        self.actor = nn.Sequential(
            layer_init(
                nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, envs.single_action_space.n), std=0.01),
        )

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)


def make_env(gym_id, seed, idx, capture_video, run_name):
    def thunk():
        env = gym.make(gym_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if capture_video:
            if idx == 0:
                env = gym.wrappers.RecordVideo(env, f"../../videos/{run_name}")
        env.seed(seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    return thunk


if __name__ == '__main__':
    args = parse_args()

    # setting run name
    run_name = f"{args.gym_id}__{args.exp_name}__{args.seed}__{int(time.time())}"

    # setting seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    # setting device: cpu or cuda
    device = torch.device(
        "cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.gym_id, args.seed + i, i, args.capture_video, run_name)
         for i in range(args.num_envs)]
    )
    assert isinstance(envs.single_action_space,
                      gym.spaces.Discrete), "only discrete action space is supported"

    agent = Agent(envs=envs).to(device=device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # storage setup
    obs = torch.zeros((args.num_steps, args.num_envs) +
                      envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) +
                          envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # start the game
    global_step = 0
    start_time = time.time()
    next_obs = torch.Tensor(envs.reset()).to(device)
    next_done = torch.zeros(args.num_envs).to(device)
    num_updates = args.total_timesteps // args.batch_size
    
    for update in range(1, num_updates + 1):
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow
        
        for step in range(0, args.num_steps):
            global_step += 1 * args.num_envs
            obs[step] = next_obs
            dones[step] = next_done
        
        
        with torch.no_grad():
            action, logprob, _, value = agent.get_action_and_value(next_obs)
            values[step] = value.flatten()
        actions[step] = action
        logprobs[step] = logprob

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, reward, done, info = envs.step(action.cpu().numpy())
        rewards[step] = torch.tensor(reward).to(device).view(-1)
        next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(done).to(device)

       