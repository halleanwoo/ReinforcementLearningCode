import tensorflow as tf
import numpy as np
import collections
import gym
import random
import tensorflow.contrib.layers as layers

ENV = "CartPole-v0"

MEMORY_SIZE = 10000
EPISODES = 500
MAX_STEP = 500
BATCH_SIZE = 32
UPDATE_PERIOD = 200  # update target network parameters


##built class for the DQN
class DeepQNetwork():
    def __init__(self , env , sess=None , gamma = 0.8, epsilon = 0.8 ):
        self.gamma = gamma
        self.epsilon = epsilon
        self.action_dim = env.action_space.n
        self.state_dim = env.observation_space.shape[0]
        self.network()
        self.sess = sess
        self.sess.run(tf.global_variables_initializer())
        tf.summary.FileWriter("DQN/summaries" , sess.graph )
        
    # net_frame using for creating Q & target network
    def net_frame(self , hiddens, inpt, num_actions, scope, reuse=None):
        with tf.variable_scope(scope, reuse=reuse):
            out = inpt  
            for hidden in hiddens:
                out = layers.fully_connected(out,  num_outputs=hidden, activation_fn=tf.nn.relu)
            out = layers.fully_connected(out, num_outputs=num_actions, activation_fn=None) 
            return out
        
    # create q_network & target_network     
    def network(self):       
        # q_network
        self.inputs_q = tf.placeholder(dtype = tf.float32 , shape = [None , self.state_dim] , name = "inputs_q")
        scope_var = "q_network"    
        self.q_value = self.net_frame([64] , self.inputs_q , self.action_dim , scope_var , reuse = True )
            
        # target_network
        self.inputs_target = tf.placeholder(dtype = tf.float32 , shape = [None , self.state_dim] , name = "inputs_target")
        scope_tar = "target_network"    
        self.q_target = self.net_frame([64] , self.inputs_target , self.action_dim , scope_tar )
               
        with tf.variable_scope("loss"):
#            #【方案一】
#             self.target = tf.placeholder(dtype = tf.float32 , shape = [None , self.action_dim] , name = "target")
#             self.loss = tf.reduce_mean( tf.square(self.q_value - self.target))
            #【方案二】
            self.action = tf.placeholder(dtype = tf.int32 , shape = [ None ] , name = "action")
            action_one_hot = tf.one_hot(self.action , self.action_dim )
            q_action = tf.reduce_sum( tf.multiply(self.q_value , action_one_hot) , axis = 1 ) 
            
            self.target =  tf.placeholder(dtype = tf.float32 , shape =  [None ] , name = "target")
            self.loss = tf.reduce_mean( tf.square(q_action - self.target))

        with tf.variable_scope("train"):
            optimizer = tf.train.RMSPropOptimizer(0.001)
            self.train_op = optimizer.minimize(self.loss)    
    
    # training
    def train(self , state , reward , action , state_next , done):
        q , q_target = self.sess.run([self.q_value , self.q_target] , 
                                     feed_dict={self.inputs_q : state , self.inputs_target : state_next } )
#         #【方案一】
#         target = reward + self.gamma * np.max(q_target , axis = 1)*(1.0 - done)
        
#         self.reform_target = q.copy()
#         batch_index = np.arange(BATCH_SIZE , dtype = np.int32)
#         self.reform_target[batch_index , action] = target
            
#         loss , _ = self.sess.run([self.loss , self.train_op] , feed_dict={self.inputs_q: state , self.target: self.reform_target} )

        #【方案二】
        q_target_best = np.max(q_target , axis = 1)
        q_target_best_mask = ( 1.0 - done) * q_target_best
        
        target = reward + self.gamma * q_target_best_mask
        
        loss , _ = self.sess.run([self.loss , self.train_op] , 
                                 feed_dict={self.inputs_q: state , self.target:target , self.action:action} )    
    # chose action
    def chose_action(self , current_state):
        current_state = current_state[np.newaxis , :]  #*** array dim: (xx,)  --> (1 , xx) ***
        q = self.sess.run(self.q_value , feed_dict={self.inputs_q : current_state} )
        
        # e-greedy
        if np.random.random() < self.epsilon:
            action_chosen = np.random.randint(0 , self.action_dim)
        else:
            action_chosen = np.argmax(q)
        
        return action_chosen
         
    #upadate parmerters
    def update_prmt(self):
        q_prmts = tf.get_collection( tf.GraphKeys.GLOBAL_VARIABLES ,   "q_network"  )
        target_prmts = tf.get_collection( tf.GraphKeys.GLOBAL_VARIABLES,  "target_network"  )
        self.sess.run( [tf.assign(t , q)for t,q in zip(target_prmts , q_prmts)])  #***
        print("updating target-network parmeters...")
        
    def decay_epsilon(self):
        if self.epsilon > 0.03:
            self.epsilon = self.epsilon - 0.02

# memory for momery replay
memory = []
Transition = collections.namedtuple("Transition" , ["state", "action" , "reward" , "next_state" , "done"])

if __name__ == "__main__":
    env = gym.make(ENV)
    with tf.Session() as sess:
        DQN = DeepQNetwork(env , sess )
        update_iter = 0
        step_his = []
        for episode in range(EPISODES):
            state = env.reset()
            env.render() 
            reward_all = 0
            #training
            for step in range(MAX_STEP):
                action = DQN.chose_action(state)
                next_state , reward , done , _ = env.step(action)
                reward_all += reward 

                if len(memory) > MEMORY_SIZE:
                    memory.pop(0)
                memory.append(Transition(state, action , reward , next_state , float(done)))

                if len(memory) > BATCH_SIZE * 4:
                    batch_transition = random.sample(memory , BATCH_SIZE)
                    #***
                    batch_state, batch_action, batch_reward, batch_next_state, batch_done = map(np.array , zip(*batch_transition))  
                    DQN.train(state = batch_state ,
                              reward = batch_reward , 
                              action = batch_action , 
                              state_next = batch_next_state,
                              done = batch_done
                             )
                    update_iter += 1

                if update_iter % UPDATE_PERIOD == 0:
                    DQN.update_prmt()
                
                if update_iter % 200 == 0:
                    DQN.decay_epsilon()

                if done:
                    print("[episode = {} ] step = {}".format(episode , step))
                    break
                    
                state = next_state
            