import torch
import numpy as np
import torch.nn as nn
from torch.nn.functional import gumbel_softmax, softmax
import torch.optim as optim
import torch.multiprocessing as mp
import os


# Hyperparameters
#hidden_dim = 10
#state_dim = 0
#action_dim = 1

class RewardModel(nn.Module):
  def __init__(self, state_dim, action_dim, hidden_dim):
    super(RewardModel, self).__init__()
    self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
    self.fc2 = nn.Linear(hidden_dim, hidden_dim)
    self.fc3 = nn.Linear(hidden_dim, 1)

  def forward(self, x):
    #x = torch.cat((state, action), dim=1)
    x = torch.relu(self.fc1(x))
    x = torch.relu(self.fc2(x))
    x = self.fc3(x)
    x = torch.sigmoid(x)
    return x

# Behaviour Cloning Policy Model:
class BCPolicy(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(BCPolicy, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)


    def forward(self, x):
        # x = torch.cat((state, action), dim=1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        #x = torch.sigmoid(x)
        return x

class BCPolicyDiscrete(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(BCPolicyDiscrete, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)


    def forward(self, x):
        # x = torch.cat((state, action), dim=1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        x = softmax(x)
        return x
    

class LLMPolicyAdapter(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(LLMPolicyAdapter, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)