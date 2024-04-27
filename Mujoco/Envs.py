import numpy as np
from gym import spaces
import gym

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