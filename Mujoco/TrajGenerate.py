import numpy as np
from Envs import MujocoEnv
from gym import spaces
import gym
from stable_baselines3 import PPO
import time
import torch

# Generate Trajectories with RandomPolicy
def TrajRandomPolicy(seed=33, env_name='Swimmer', num=100):

    env = MujocoEnv(env_name)

    env.seed(seed)
    episode_reward = []
    action_rollout = []
    state_rollout = []
    for tr in range(num):
        env.seed(seed)
        obs = env.reset()

        total_reward = 0
        reward_record = []
        action_record = []
        state_record = []

        state_record.append(obs)

        for _ in range(10):
            if env_name == 'Swimmer':
                action = np.random.uniform(-1, 1, (2))
            elif env_name == 'HalfCheetah':
                action = np.random.uniform(-1, 1, (6))

            obs, reward, done, _ = env.step(action)
            total_reward += reward
            reward_record.append(total_reward)

            action_record.append(action)
            state_record.append(obs)

            # if done:
            # obs = env.reset()
        episode_reward.append(total_reward)
        action_rollout.append(action_record)
        state_rollout.append(state_record)

    #print("Episode Reward:", episode_reward)
    print('Avg Reward:', np.mean(np.array(episode_reward)))
    print('Std:', np.std(np.array(episode_reward)))

    action_rollout = np.array(action_rollout)
    state_rollout = np.array(state_rollout)
    episode_reward = np.array(episode_reward)

    return action_rollout, state_rollout, episode_reward


# Generate Trajectories with RandomPolicy
def TrajExpertPolicy(seed=33, env_name='Swimmer', num=100, episode=100):
    env = MujocoEnv(env_name)

    expert_agent = PPO('MlpPolicy', env, batch_size=50, n_steps=10, verbose=1)
    expert_agent.learn(10*episode)

    env.seed(seed)
    episode_reward = []
    action_rollout = []
    state_rollout = []
    for tr in range(num):
        env.seed(seed)
        obs = env.reset()

        total_reward = 0
        reward_record = []
        action_record = []
        state_record = []

        state_record.append(obs)

        for _ in range(10):
            action = expert_agent.predict(obs)
            obs, reward, done, _ = env.step(action)
            total_reward += reward
            reward_record.append(total_reward)

            action_record.append(action)
            state_record.append(obs)

            # if done:
            # obs = env.reset()
        episode_reward.append(total_reward)
        action_rollout.append(action_record)
        state_rollout.append(state_record)

    #print("Episode Reward:", episode_reward)
    print('Avg Reward:', np.mean(np.array(episode_reward)))
    print('Std:', np.std(np.array(episode_reward)))

    action_rollout = np.array(action_rollout)
    state_rollout = np.array(state_rollout)
    episode_reward = np.array(episode_reward)

    return action_rollout, state_rollout, episode_reward