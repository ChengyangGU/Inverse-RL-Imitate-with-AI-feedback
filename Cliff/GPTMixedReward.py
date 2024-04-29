import numpy as np
import time
import torch
import torch.nn as nn
import torch.optim as optim
from Envs import CliffWalkingEnv
from Expert_Traj import Cliff_Expert_Action
from GPTPreLabels import GPT_pre
from Models import RewardModel
from BehaviourCloning import StateTrajGenerate, ReformAction

# CrossEntropy Loss for 0-1 Preference Labels
def MixedLoss(R1, R2, L1, L2, Expert_Reward, w):

    p1 = torch.exp(R1) / (torch.exp(R1) + torch.exp(R2))
    p2 = torch.exp(R2) / (torch.exp(R1) + torch.exp(R2))

    #e1 = torch.exp(Expert_Reward) / (torch.exp(Expert_Reward) + torch.exp(R1))
    #e2 = torch.exp(Expert_Reward) / (torch.exp(Expert_Reward) + torch.exp(R2))


    #loss = -(L1 * torch.log(p1) + L2 * torch.log(p2)) - torch.log(e1) - torch.log(e2)
    loss = -(L1 * torch.log(p1) + L2 * torch.log(p2)) - w * Expert_Reward
    #print('R1R2Check:', R1, R2)
    #print('p1p2Check:', p1, p2)
    #print('e1e2Check:', e1, e2)
    #print('PreLossCheck:', -(L1 * torch.log(p1) + L2 * torch.log(p2)))
    #print('ExpertReward:', Expert_Reward)

    return loss



if __name__ == '__main__':

    action_dataset = np.load('random_action.npy', allow_pickle=True)
    state_dataset = np.load('random_state.npy', allow_pickle=True)

    # (num, epi_length, state_dim=1)
    exp_action_rollout = ReformAction(action_traj=Cliff_Expert_Action, epi_len=100)
    # (num, epi_length, action_dim=1)
    exp_state_rollout = StateTrajGenerate(action_traj=Cliff_Expert_Action, epi_len=100)

    action_rollout = np.ones((100, 100))
    state_rollout = np.zeros((100, 100))


    expert_action = exp_action_rollout[0]
    expert_action = torch.tensor(expert_action, dtype=torch.float32)
    expert_action = expert_action.unsqueeze(1)

    expert_state = exp_state_rollout[0]
    expert_state = torch.tensor(expert_state, dtype=torch.float32)
    expert_state = expert_state.unsqueeze(1)

    expert_input = torch.concat((expert_state, expert_action), dim=1)
    #expert_input = torch.tensor(expert_input, dtype=torch.float32)
    print(expert_input.shape)


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
    imitate_weight = 0.02
    loss_fn = MixedLoss
    optimizer = optim.Adam(reward_model.parameters(), lr=learning_rate)

    # Record Loss Curve
    loss_traj = []
    for epoch in range(num_epochs):
        loss = 0
        expert_reward = reward_model(expert_input).sum()
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
            loss += loss_fn(R1, R2, L1, L2, expert_reward, w=imitate_weight)

        #print('PreLoss:', loss)

        #expert_lead = - torch.exp(expert_input)

        #print('ExpertReward:', expert_reward)
        #loss = loss - imitate_weight * expert_reward
        loss.backward()
        optimizer.step()
        print(f"Epoch: {epoch}, Loss: {loss.item()}")
        loss_traj.append(loss.item())

    filename = f"./ModelSave/MixReward.pth"
    torch.save(reward_model, filename)
