import argparse
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
    parser = argparse.ArgumentParser()
    parser.add_argument('--hidden_dim', type=int, default=100)
    parser.add_argument('--epi_len', type=int, default=100)
    parser.add_argument('--num_epoches', type=int, default=1000)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--path_of_expert_act', type=str, default='CliffTraj/Expert_action.npy')
    parser.add_argument('--path_of_expert_state', type=str, default='CliffTraj/Expert_state_2_dim.npy')
    parser.add_argument('--path_of_llm_act', type=str, default='CliffTraj/LLM_action.npy')
    parser.add_argument('--path_of_llm_state', type=str, default='CliffTraj/LLM_state_2_dim.npy')
    
    args = parser.parse_args()
    epi_len = args.epi_len

    # (num, epi_length, state_dim=1)
    cliff_expert_action = np.load(args.path_of_expert_act, allow_pickle=True)
    cliff_expert_action = cliff_expert_action.reshape(cliff_expert_action.shape[0], cliff_expert_action.shape[1], 1)
    # (num', epi_length, action_dim=1)
    cliff_llm_action = np.load(args.path_of_llm_act, allow_pickle=True)
    cliff_llm_action = cliff_llm_action.reshape(cliff_llm_action.shape[0], cliff_llm_action.shape[1], 1)
    # (num, epi_length, space_dim=2)
    cliff_expert_state = np.load(args.path_of_expert_state, allow_pickle=True)
    # (num', epi_length, space_dim=2)
    cliff_llm_state = np.load(args.path_of_llm_state, allow_pickle=True)    

    # Train Parameters
    policy_model = BCPolicy(state_dim=1, action_dim=1, hidden_dim=args.hidden_dim)
    learning_rate = args.learning_rate
    num_epoches = args.num_epoches
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