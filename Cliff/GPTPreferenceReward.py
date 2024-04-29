import numpy as np
import time
import torch
import torch.nn as nn
import torch.optim as optim
from Envs import CliffWalkingEnv
from Expert_Traj import Cliff_Expert_Action
from GPTPreLabels import GPT_pre
from Models import RewardModel


# CrossEntropy Loss for 0-1 Preference Labels
def CELoss(R1, R2, L1, L2):

  p1 = torch.exp(R1) / (torch.exp(R1) + torch.exp(R2))
  p2 = torch.exp(R2) / (torch.exp(R1) + torch.exp(R2))

  loss = -(L1 * torch.log(p1) + L2 * torch.log(p2))

  return loss



if __name__ == '__main__':

    action_dataset = np.load('random_action.npy', allow_pickle=True)
    state_dataset = np.load('random_state.npy', allow_pickle=True)

    action_rollout = np.ones((100, 100))
    state_rollout = np.zeros((100, 100))

    for tr in range(100):
        length = len(action_dataset[tr])
        for j in range(length):
            action_rollout[tr][j] = action_dataset[tr][j]
            state_rollout[tr][j] = state_dataset[tr][j]
        for j in range(length,100):
            action_rollout[tr][j] = 1
            state_rollout[tr][j] = 47

    # GPT Preference Labels
    gpt_labels = np.zeros((50, 2))
    for pair in range(50):
        if GPT_pre[pair] == 1:
            gpt_labels[pair][0] = 1
        else:
            gpt_labels[pair][1] = 1



    # Dataset: (pair num, pair ID, time_length, state_dim + action_dim)
    D = np.zeros((50, 2, 100, 1 + 1))

    for k in range(50):
        pair1 = 2 * k
        pair2 = 2 * k + 1

        for i in range(100):
            D[k][0][i][0] = state_rollout[pair1][i]
            D[k][1][i][0] = state_rollout[pair2][i]

            D[k][0][i][1] = action_rollout[pair1][i]
            D[k][1][i][1] = action_rollout[pair2][i]


    # Train Parameters
    reward_model = RewardModel(state_dim=1, action_dim=1, hidden_dim=10)
    learning_rate = 0.001
    num_epochs = 700
    loss_fn = CELoss
    optimizer = optim.Adam(reward_model.parameters(), lr=learning_rate)

    # Record Loss Curve
    loss_traj = []
    for epoch in range(num_epochs):
        loss = 0
        for i in range(D.shape[0]):
            cur_pair = D[i]
            reward_input_1 = cur_pair[0]
            reward_input_1 = torch.tensor(reward_input_1, dtype=torch.float32)

            reward_input_2 = cur_pair[1]
            reward_input_2 = torch.tensor(reward_input_2, dtype=torch.float32)

            R1 = reward_model(reward_input_1).sum()
            R2 = reward_model(reward_input_2).sum()

            L1 = gpt_labels[i][0]
            L2 = gpt_labels[i][1]
            loss += loss_fn(R1, R2, L1, L2)

        loss.backward()
        optimizer.step()
        print(f"Epoch: {epoch}, Loss: {loss.item()}")
        loss_traj.append(loss.item())

    filename = f"./ModelSave/PreReward.pth"
    torch.save(reward_model, filename)
