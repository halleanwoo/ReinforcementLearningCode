import gym
from atari_wrappers import *
from argument import args

def envMakeWrapper(env_name, flag_cnn=False):
    if not flag_cnn:
        print("not wrapper")
        env = gym.make(env_name)
        env.seed(args.env_seed)
        return 
        
    else:
        print("wrapper")
        env = make_atari(env_name)
        env = wrap_deepmind(env, frame_stack=True, scale=True)
        env.seed(args.env_seed)
        return env

