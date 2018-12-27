#!usr/bin/env python3
# -*- coding: utf-8 -*- 

import tensorflow as tf
import numpy as np
import gym
import random
import tensorflow.contrib.layers as layers
import os
import time
from netFrame import net_frame_mlp, net_frame_cnn_to_mlp, build_net
from memory import Memory
from utils import *
from argument import args

class DeepQNetwork4Atari():
    def __init__(self , scope_main, env, flag_double, flag_cnn, sess=None , gamma = 0.99):
        self.gamma = gamma
        self.epsilon = 1.0
        self.action_dim = env.action_space.n
        self.state_shape = [None] + list(env.observation_space.shape)   # levin soft code: [None,84,84,4] or [None, xx..]
        self.scope_main = scope_main
        self.flag_double = flag_double
        self.flag_cnn = flag_cnn
        self.network()
        self.sess = sess
        
        # self.merged = tf.summary.merge_all()
        # self.write2 = tf.summary.FileWriter("HVDQN/test1/2" , sess.graph )

        self.sess.run(tf.global_variables_initializer())
        # self.merged = tf.merge_all_summaries()
        # self.result_tensorboar
        
    # create q_network & target_network   
    def network(self): 
        self.inputs_q = tf.placeholder(dtype = tf.float32 , shape = self.state_shape, name = "inputs_q")
        scope_var = "q_network" 

        self.inputs_target = tf.placeholder(dtype = tf.float32 , shape = self.state_shape , name = "inputs_target")
        scope_tar = "target_network"

        # q_network
        if self.flag_cnn:
            self.q_value = net_frame_cnn_to_mlp(convs=[(32, 8, 4), (64, 4, 2), (64, 3, 1)], 
                                                hiddens=[512], 
                                                inpt=self.inputs_q, 
                                                num_actions=self.action_dim, 
                                                scope=scope_var,
                                                dueling=0,)
        else:
            self.q_value = net_frame_mlp([32,16] , self.inputs_q , self.action_dim , scope=scope_var )
        
        # target_network  
        if self.flag_cnn:
            self.q_target = net_frame_cnn_to_mlp(convs=[(32, 8, 4), (64, 4, 2), (64, 3, 1)], 
                                                 hiddens=[512], 
                                                 inpt=self.inputs_target, 
                                                 num_actions=self.action_dim, 
                                                 scope=scope_tar,
                                                 dueling=0,)
        else:
            self.q_target = net_frame_mlp([32,16] , self.inputs_target , self.action_dim , scope_tar )
        
        # # === test ===
        # self.q_value = build_net(self.inputs_q, scope_var)
        # self.q_target = build_net(self.inputs_target, scope_tar)

        with tf.variable_scope("loss"):
            self.action = tf.placeholder(dtype = tf.int32 , shape = [None] , name = "action")
            self.action_one_hot = tf.one_hot(self.action , self.action_dim )
            q_action = tf.reduce_sum( tf.multiply(self.q_value , self.action_one_hot) , axis = 1 ) 
            
            self.target = tf.placeholder(dtype = tf.float32 , shape =  [None] , name = "target")

            # # ----- L2Loss or huberLoss -------
            # self.loss = tf.reduce_mean( tf.square(q_action - self.target))
            self.loss = tf.reduce_mean( huber_loss(q_action - self.target))

        with tf.variable_scope("train"):
            # optimizer = tf.train.RMSPropOptimizer(args.learning_rate_Q, decay=0.99, momentum=0.0, epsilon=1e-6)    # 0.001  0.0005(better)  0.0002(net-best)
            # # optimizer = tf.train.AdamOptimizer(args.learning_rate_Q,) 
            # if args.flag_CLIP_NORM :
            #     gradients = optimizer.compute_gradients(self.loss)
            #     for i , (g, v) in enumerate(gradients):
            #         if g is not None:
            #             gradients[i] = (tf.clip_by_norm(g , 1) , v)
            #     self.train_op = optimizer.apply_gradients(gradients)
            # else:
            #     self.train_op = optimizer.minimize(self.loss)
            self.train_op = build_rmsprop_optimizer(args.learning_rate_Q, 0.99, 1e-6, 1, 'rmsprop', loss=self.loss)
               
    # training
    def train(self, state, reward, action, state_next, done, episode_return, estim_Qvalue_argmax, estim_Qvalue_expect):
        q, q_target = self.sess.run([self.q_value, self.q_target], 
                                     feed_dict={self.inputs_q: state, self.inputs_target: state_next})
        # dqn
        if not self.flag_double:
            q_target_best = np.max(q_target, axis = 1)
        # doubel dqn
        else:
            q_next = self.sess.run(self.q_value , feed_dict={self.inputs_q : state_next})
            action_best = np.argmax(q_next , axis = 1)
            action_best_one_hot = self.sess.run(self.action_one_hot, feed_dict={self.action: action_best})
            q_target_best = np.sum(q_target * action_best_one_hot, axis=1)

        q_target_best_mask = ( 1.0 - done) * q_target_best
        target = reward + self.gamma * q_target_best_mask

        loss, _ = self.sess.run([self.loss, self.train_op] , 
                                 feed_dict={self.inputs_q: state , self.target: target , self.action: action} )
            # if update_iter % SUMMARY_PERIOD == 0:
            #     result = self.sess.run(self.merged,
            #                            feed_dict={self.inputs_q: state , self.target: target , self.action: action, self.inputs_target: state} )
            #     self.write.add_summary(result, update_iter)

        return estim_Qvalue_argmax, estim_Qvalue_expect

    # chose action
    def chose_action(self , current_state):
        
        # e-greedy
        if np.random.random() < self.epsilon:
            # action_chosen = np.random.randint(0 , self.action_dim)
            action_chosen = random.randrange(self.action_dim)

        else:
            current_state = current_state[np.newaxis , :]  # *** array dim: (xx,)  --> (1 , xx) ***
            q = self.sess.run(self.q_value , feed_dict={self.inputs_q : current_state} )
            action_chosen = np.argmax(q)
        return action_chosen

    def greedy_action(self , current_state):
        current_state = current_state[np.newaxis , :]  
        # print(current_state)
        q = self.sess.run(self.q_value , feed_dict={self.inputs_q : current_state} ) 
        action_greedy = np.argmax(q)
        return action_greedy
         
    #upadate parmerters
    def update_prmt(self):
        q_prmts = tf.get_collection( tf.GraphKeys.GLOBAL_VARIABLES, self.scope_main + "/q_network"  )            
        target_prmts = tf.get_collection( tf.GraphKeys.GLOBAL_VARIABLES, self.scope_main +  "/target_network"  )    
        self.sess.run( [tf.assign(t , q)for t,q in zip(target_prmts , q_prmts)])  #***
        print("===updating target-network parmeters===")
        
    def decay_epsilon(self, episode, SUM_EP):
        episode = episode * 1.0
        faster_factor = 1.0  # 1.2 ... 2...
        if self.epsilon > 0.1:
            self.epsilon = 1 - episode * faster_factor / SUM_EP
        # print("epsilon:............................",self.epsilon)

