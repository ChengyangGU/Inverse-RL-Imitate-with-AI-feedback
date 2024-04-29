import numpy as np


GPT_pre = [2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1,
           2, 1, 2, 1, 2, 1, 2, 1, 2, 1]
GPT_pre = np.array(GPT_pre)





if __name__ == '__main__':
    episode_reward = np.load('random_reward.npy', allow_pickle=True)
    True_pre = np.zeros(50)

    for k in range(50):
        print('Pair', k+1)
        print('Rollout 1 True Reward:', episode_reward[2*k])
        print('Rollout 2 True Reward:', episode_reward[2*k+1])

        if episode_reward[2*k] >= episode_reward[2*k+1]:
            True_pre[k] = 1
        else:
            True_pre[k] = 2

    count = 0
    for k in range(50):
        if GPT_pre[k] == True_pre[k]:
            count += 1

    print('GPT Choice Correct Rate:', count/50*100, '%')

    print('Average True Reward:', np.mean(episode_reward))
    print('STD True Reward:', np.std(episode_reward))