import numpy as np
import gym
import random
from envWrapper import envMakeWrapper

def testQGame(env_name, agent, episode, test_reward_all_his, test_every_average1, test_every_each10, flag_cnn, max_step=10000, test_times=10):
    # print("testing_q...")
    env1 = envMakeWrapper(env_name, flag_cnn)
    if len(test_reward_all_his) is 0:
        test_reward_all = 0
    else:
        test_reward_all = test_reward_all_his[-1]
    reward_all_episode = 0
    for i in range(test_times):
        # env1.render()
        reward_episode = 0
        state = env1.reset()                  
        for step in range(max_step):
            action = agent.greedy_action(state)
            next_state , reward , done , _ = env1.step(action)
            reward_episode += reward
            if done:
                print("times: %2d -- step: %3d -- reward: %d"%(i, step, reward_episode))

                test_reward_all += reward_episode
                reward_all_episode += reward_episode
                test_reward_all_his.append(test_reward_all)
                test_every_each10.append(reward_episode)
                break
            state = next_state
    test_average = reward_all_episode / test_times
    test_every_average1.append(test_average)

    print("test_times: %d , per_reward: %.4f"%( test_times , test_average)  )
    return test_reward_all_his, test_every_average1, test_every_each10
