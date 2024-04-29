from Envs import CliffWalkingEnv
import numpy as np

def GenerateRandomTraj(num=100):
    env = CliffWalkingEnv()

    episode_reward = []
    action_rollout = []
    state_rollout = []

    for tr in range(num):
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
            action = np.random.randint(0, 4)
            obs, reward, done, _ = env.step(action)
            # print('Action:', action)
            # print('State:', obs)
            # print('Reward:', total_reward)
            total_reward += reward
            reward_record.append(total_reward)

            action_record.append(action)
            state_record.append(obs)
            step += 1

        episode_reward.append(total_reward)
        action_rollout.append(action_record)
        state_rollout.append(state_record)

    print("Episode Reward:", episode_reward)
    print('Avg Reward:', np.mean(np.array(episode_reward)))
    print('Std:', np.std(np.array(episode_reward)))

    return action_rollout, state_rollout, episode_reward

if __name__ == '__main__':
    action_rollout, state_rollout, episode_reward = GenerateRandomTraj(num=100)

    action_rollout = np.array(action_rollout)
    state_rollout = np.array(state_rollout)

    print('Max Reward:', np.max(np.array(episode_reward)))
    np.save('random_action',action_rollout)
    np.save('random_state',state_rollout)
    np.save('random_reward', episode_reward)

