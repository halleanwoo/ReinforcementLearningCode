import tensorflow as tf
import numpy as np
import gym
import os
import time

from testModel import testQGame # testQ, 
from netFrame import net_frame_mlp, net_frame_cnn_to_mlp
from utils import *
from envWrapper import envMakeWrapper
from DQNModel import DeepQNetwork4Atari
from memory import Memory

from argument import args
from logger import Logger


def training(agent, env, flag_using_heu, file = None):
    update_iter = 0
    reward_every = []
    Max_return = 0
    reward_all = 0

    reward_all_his = []
    estim_Qvalue_argmax = []
    estim_Qvalue_expect = []

    test_reward_all = []
    test_every_average1_Q = []
    test_every_each10_Q = []

    test_reward_H = []
    test_every_H = []
    

    reward_episode = 0   #!!!!!
    ep_flag = 0

    for episode in range(args.epoches):
        state = env.reset()
        # env.render() 

        if episode != 0 and (episode + 1) % args.test_period == 0 and update_iter > args.observe_step:
            test_reward_all, test_every_average1_Q, test_every_each10_Q = testQGame(args.env_name, agent, episode, test_reward_all, test_every_average1_Q, test_every_each10_Q, args.flag_cnn, args.max_step)  

        #training
        for step in range(args.max_step):
            action = agent.chose_action(state)
            next_state , reward , done , _ = env.step(action)
            update_iter += 1
            reward_episode += reward
            # tf.summary.scalar('reward_all',reward_all)
            
            memory.store(state, action , reward , next_state , done)

            if memory.size() >  args.observe_step:              # [TODO-why: observe so much?]
                if update_iter % args.frame_skip == 0:          # levin: without testing    [TODO-why: significant?]
                #***
                    batch_state, batch_action, batch_reward, batch_next_state, batch_done, batch_return = memory.batchSample(args.batch_size)
                    estim_Qvalue_argmax, estim_Qvalue_expect = agent.train(state=batch_state ,
                                                                           reward= batch_reward , 
                                                                           action = batch_action , 
                                                                           state_next = batch_next_state,
                                                                           done = batch_done,
                                                                           episode_return = batch_return,
                                                                           estim_Qvalue_argmax=estim_Qvalue_argmax,
                                                                           estim_Qvalue_expect=estim_Qvalue_expect,

                                                                         )
                    # bat_s = np.mean(batch_state, axis=0)
                    # bat_a = np.mean(batch_action, axis=0)

                # if flag_summary:
                #     agent.write_summary(state = batch_state ,
                #                             reward = batch_reward , 
                #                           action = batch_action , 
                #                           state_next = batch_next_state,
                #                           done = batch_done,
                #                           episode_return = batch_return,
                #                           summary_iter = update_iter,
                #                           reward_all = reward_all,
                #                           flag_using_heu = flag_using_heu
                #                          )


            if update_iter > args.observe_step and  update_iter % args.update_period == 0:
                agent.update_prmt()
                
            if update_iter > args.observe_step and update_iter % 100 == 0 and update_iter != 0:
                if update_iter > args.observe_step:
                    agent.decay_epsilon(update_iter - args.observe_step, args.explore_step)

            # episode or epoch, if episode: done --> break
            if done:  
                if args.flag_done_break:                       
                    print(" epoch:%3d  ,  step: %3d   ,  epsilon: %.3f   ,  reward: %d"%(episode, step, agent.epsilon, reward_episode))
                    # file = open(file_path,'w')
                    # file.write(" episode:%3d  ,  step:%3d   ,  reward: %d"%(episode, step, reward_episode))
                    # file.write("\n")d
                    reward_every.append(reward_episode)
                    # break
                   
                    reward_all += reward_episode
                    reward_all_his.append(reward_all)
                    reward_episode = 0
                    # next_state = env.reset()
                    break
            
                else:
                    next_state = env.reset()        # TODO: doubt --- seems wrong?

            if step == args.max_step - 1:
                print(" epoch:%3d  ,  step: %3d   ,  epsilon: %.3f   ,  reward: %d"%(episode, step, agent.epsilon, reward_episode))
                # file = open(file_path,'w')
                # file.write(" episode:%3d  ,  step:%3d   ,  reward: %d"%(episode, step, reward_episode))
                # file.write("\n")
                reward_every.append(reward_episode)
                # break
               
                reward_all += reward_episode
                reward_all_his.append(reward_all)
                reward_episode = 0
                
                
            state = next_state
    return reward_all_his, reward_every, estim_Qvalue_argmax, estim_Qvalue_expect, test_reward_all, test_every_average1_Q, test_every_each10_Q, test_reward_H, test_every_H


if __name__ == "__main__":
    set_random_seed(args.seed)

    env_set = ["BreakoutNoFrameskip-v4", "PongNoFrameskip-v4"]
    if args.env_name in env_set:
        args.flag_cnn = True
    else:
        args.flag_cnn = False

    # set dir_path
    if args.flag_save:
        dir_path = str(int(time.time()))
        path = str(os.getcwd())
        new_path = path  + '/' + 'New_' + args.env_name + dir_path
        os.makedirs(new_path)

    memory = Memory(args.memory_size, flag_piexl=args.flag_cnn)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    # config.gpu_options.per_process_gpu_memory_fraction = 0.4
    print(config)

    with tf.Session(config=config) as sess2:
        env_1 = envMakeWrapper(args.env_name, args.flag_cnn)
        scope_name=str(time.time())
        with tf.variable_scope(scope_name):
            print("")
            print("*******************")
            print("double-DQN" if args.flag_double_dqn else "DQN")
            print("*******************")
            DQN = DeepQNetwork4Atari(scope_name, env_1 , args.flag_double_dqn, args.flag_cnn, sess2)
            reward_all_his, reward_every, estim_Qvalue_argmax, estim_Qvalue_expect, test_reward_all, test_every_average1_Q, test_every_each10_Q, test_reward_H, test_every_H = training(DQN, env_1, args.flag_double_dqn)
            # write(args.env_name, new_path, flag_using_expect, reward_all_his, reward_every, estim_Qvalue_argmax, estim_Qvalue_expect, test_reward_all, test_every_average1_Q, test_every_each10_Q, test_reward_H, test_every_H)
