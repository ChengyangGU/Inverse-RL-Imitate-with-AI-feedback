import numpy as np
import time
import torch
import torch.nn as nn
import torch.optim as optim
from copy import deepcopy
from gym import spaces
import gym
import torch.multiprocessing as mp
from Envs import CliffWalkingEnv
from Expert_Traj import Cliff_Expert_Action
from GPTPreLabels import GPT_pre
from Models import RewardModel
from stable_baselines3 import PPO

class CliffWalkingEnvTrainReward(gym.Env):
    ''' Cliff Walking Environment

          See the README.md file from https://github.com/caburu/gym-cliffwalking
      '''

    # There is no renderization yet
    # metadata = {'render.modes': ['human']}

    def observation(self, state):
        return state[0] * self.cols + state[1]

    def __init__(self, reward_model):
        self.rows = 4
        self.cols = 12
        self.start = [3, 0]
        self.goal = [3, 11]
        self.current_state = None

        # Episode Truncation length
        self.T = 100

        # Current Timestep
        self.timestep = 0

        # There are four actions: up, down, left and right
        self.action_space = spaces.Discrete(4)

        # observation is the x, y coordinate of the grid
        self.observation_space = spaces.Discrete(self.rows * self.cols)

        # Trained Reward Model
        self.reward_model = reward_model

    def step(self, action):

        self.timestep += 1
        new_state = deepcopy(self.current_state)

        if action == 1:  # right
            new_state[1] = min(new_state[1] + 1, self.cols - 1)
        elif action == 0:  # up
            new_state[0] = max(new_state[0] - 1, 0)
        elif action == 3:  # left
            new_state[1] = max(new_state[1] - 1, 0)
        elif action == 2:  # down
            new_state[0] = min(new_state[0] + 1, self.rows - 1)
        else:
            raise Exception("Invalid action.")
        self.current_state = new_state

        state_idx = self.observation(self.current_state)
        input_tensor = np.concatenate((np.array([state_idx]), np.array([action])), axis=0)
        input_tensor = torch.tensor(input_tensor, dtype=torch.float32)
        #print(input_tensor)
        input_tensor.unsqueeze(0)

        #print(input_tensor)
        reward = self.reward_model(input_tensor)
        reward = reward.item()

        is_terminal = False
        if self.current_state[0] == 3 and self.current_state[1] > 0:
            if self.current_state[1] < self.cols - 1:
                #reward = -100.0
                self.current_state = deepcopy(self.start)
            else:
                is_terminal = True
                #reward = 100

        if self.timestep >= self.T:
            is_terminal = True

        return self.observation(self.current_state), reward, is_terminal, {}

    def reset(self):
        self.current_state = self.start
        self.timestep = 0
        return self.observation(self.current_state)

    def render(self, mode='human'):
        pass

    def close(self):
        pass

def PerformEvaluate(agent):
    episode_reward = []
    action_rollout = []
    state_rollout = []

    env = CliffWalkingEnv()
    for tr in range(100):
        # print('Traj:', tr+1)
        obs = env.reset()
        total_reward = 0
        reward_record = []
        action_record = []
        state_record = []
        done = False
        step = 0
        while done == False:
            # print('Step:', step)
            action = agent.predict(obs)
            action = action[0]
            obs, reward, done, _ = env.step(action)
            # print('Action:', action)
            # print('State:', obs)
            # print('Reward:', total_reward)
            total_reward += reward
            reward_record.append(total_reward)

            action_record.append(action)
            state_record.append(obs)
            step += 1

            # if done:
            # obs = env.reset()
        episode_reward.append(total_reward)
        action_rollout.append(action_record)
        state_rollout.append(state_record)

    print("Episode Reward:", episode_reward)
    print('Avg Reward:', np.mean(np.array(episode_reward)))
    print('Std:', np.std(np.array(episode_reward)))

    return action_rollout, state_rollout, episode_reward


if __name__ == '__main__':

    filename = f"./ModelSave/MixReward.pth"
    reward_model = RewardModel(state_dim=1, action_dim=1, hidden_dim=10)
    reward_model = torch.load(filename)

    #Train RL Agent
    env_train = CliffWalkingEnvTrainReward(reward_model=reward_model)
    agent = PPO('MlpPolicy', env_train, batch_size=50, n_steps=100, verbose=1)
    episode = 400
    agent.learn(total_timesteps=episode * 100)

    # Evaluate
    _, _, _ = PerformEvaluate(agent=agent)