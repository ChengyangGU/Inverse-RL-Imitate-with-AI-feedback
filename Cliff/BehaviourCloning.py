import numpy as np
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
from Envs import CliffWalkingEnv
from Expert_Traj import Cliff_Expert_Action
from Models import BCPolicy

# Generate State Rollout
def StateTrajGenerate(action_traj, epi_len=100):
    env = CliffWalkingEnv()

    # Generate Action
    state_rollout = np.ones((len(action_traj), epi_len))

    for i in range(len(action_traj)):
        obs = env.reset()
        action_list = action_traj[i]
        state_rollout[i][0] = obs
        for j in range(len(action_list)):
            action = action_list[j]
            obs, reward, done, _ = env.step(action)

            # Terminal Branch
            if done == False:
                state_rollout[i][j+1] = obs
            else:
                state_rollout[i][j+1:] = 47

    return state_rollout

# Reform Action to One-hot
def ReformAction(action_traj, epi_len=100):

    # Generate Action
    action_rollout = np.ones((len(action_traj), epi_len))
    for i in range(len(action_traj)):
        action_list = action_traj[i]
        for j in range(len(action_list)):
            action = action_list[j]
            action_rollout[i][j] = action

    return action_rollout

# Reform Action to One-hot
def ReformActionOneHot(action_traj, epi_len=100):

    # Generate Action
    action_rollout = np.zeros((len(action_traj), epi_len, 4))
    for i in range(len(action_traj)):
        action_list = action_traj[i]
        for j in range(len(action_list)):
            action = action_list[j]
            #print(action)
            if action == 0:
                action_rollout[i][j][0] = 1
            elif action == 1:
                action_rollout[i][j][1] = 1
            elif action == 2:
                action_rollout[i][j][2] = 1
            elif action == 3:
                action_rollout[i][j][3] = 1

        for j in range(len(action_list), epi_len):
            action_rollout[i][j][1] = 1

    return action_rollout


# Test Agent Performance:
def AgentTest(model):
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
            obs = torch.tensor([obs],dtype=torch.float32)
            action = model(obs)
            action = np.round(action.item())
            if action > 3:
                action = 3
            elif action < 0:
                action = 0
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

    # (num, epi_length, state_dim=1)
    action_rollout = ReformAction(action_traj=Cliff_Expert_Action, epi_len=100)
    # (num, epi_length, action_dim=1)
    state_rollout = StateTrajGenerate(action_traj=Cliff_Expert_Action, epi_len=100)
    # (num, epi_length, action_dim)
    action_rollout_one_hot = ReformActionOneHot(action_traj=Cliff_Expert_Action, epi_len=100)


    # Output (Label)
    action_rollout = torch.tensor(action_rollout, dtype=torch.float32)
    action_rollout = action_rollout.flatten()
    action_rollout = action_rollout.reshape(-1, 1)

    # Input (State)
    state_rollout = torch.tensor(state_rollout, dtype=torch.float32)
    state_rollout = state_rollout.flatten()
    state_rollout = state_rollout.reshape(-1, 1)


    # Train Parameters
    policy_model = BCPolicy(state_dim=1, action_dim=1, hidden_dim=100)
    learning_rate = 0.001
    num_epoches = 1000
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(policy_model.parameters(), lr=learning_rate)

    # Record Loss Curve
    loss_traj = []
    for epoch in range(num_epoches):
        input = state_rollout
        pred_policy = policy_model(input)
        expert_policy = action_rollout
        loss = loss_fn(pred_policy, expert_policy)
        # loss = MixedLoss(reward_train, labels, expert)

        loss.backward()
        optimizer.step()
        if (epoch % 5) == 0:
            print(f"Epoch: {epoch}, Loss: {loss.item()}")
        loss_traj.append(loss.item())

    # Test
    cur_state = torch.tensor([13], dtype=torch.float32)
    print(policy_model(cur_state))

    # Test Agent
    action_record, state_record, reward_record = AgentTest(policy_model)

    print(state_record[0])
    print(action_record[0])