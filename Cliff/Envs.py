import numpy as np
from gym import spaces
import gym
from copy import deepcopy

# Current Support Environment: HalfCheetah and Swimmer
class MujocoEnv(gym.Env):
    def __init__(self, EnvName):

        # Time Horizon
        self.T = 10

        # Define the action and observation spaces
        self.max_u = 1
        self.max_x = np.inf

        self.timestep = 0  # Current timestep

        if EnvName == 'Swimmer':
            self.env_base = gym.make('Swimmer-v4', exclude_current_positions_from_observation=False)
            self.action_space = spaces.Box(low=-self.max_u, high=self.max_u, shape=(2,), dtype=np.float32)
            self.observation_space = spaces.Box(low=-self.max_x, high=self.max_x, shape=(10,), dtype=np.float32)
        elif EnvName == 'HalfCheetah':
            self.env_base = gym.make('HalfCheetah-v4', exclude_current_positions_from_observation=False)
            self.action_space = spaces.Box(low=-self.max_u, high=self.max_u, shape=(6,), dtype=np.float32)
            self.observation_space = spaces.Box(low=-self.max_x, high=self.max_x, shape=(18,), dtype=np.float32)
        # Default = Swimmer
        else:
            self.env_base = gym.make('Swimmer-v4', exclude_current_positions_from_observation=False)
            self.action_space = spaces.Box(low=-self.max_u, high=self.max_u, shape=(2,), dtype=np.float32)
            self.observation_space = spaces.Box(low=-self.max_x, high=self.max_x, shape=(10,), dtype=np.float32)
        # Initialize
        self.state = self.env_base.reset()

    def reset(self):
        # Reset to start [0, 0]
        self.timestep = 0

        self.state = self.env_base.reset()
        return self.state

    def step(self, action):

        self.timestep += 1
        self.state, _, _, _ = self.env_base.step(action)

        reward = self._get_reward(action)

        done = (self.timestep >= self.T)

        return self.state, reward, done, {}

    def _get_reward(self, action):
        _, reward, _, _ = self.env_base.step(action)
        return reward

    def render(self, mode='human'):
        pass


class CliffWalkingEnv(gym.Env):
    ''' Cliff Walking Environment

          See the README.md file from https://github.com/caburu/gym-cliffwalking
      '''

    # There is no renderization yet
    # metadata = {'render.modes': ['human']}

    def observation(self, state):
        return state[0] * self.cols + state[1]

    def __init__(self):
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

        reward = -1.0
        is_terminal = False
        if self.current_state[0] == 3 and self.current_state[1] > 0:
            if self.current_state[1] < self.cols - 1:
                reward = -100.0
                self.current_state = deepcopy(self.start)
            else:
                is_terminal = True
                reward = 100

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