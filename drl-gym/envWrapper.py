import gym
from atari_wrappers import *

def envMakeWrapper(env_name, flag_cnn=False):
    if not flag_cnn:
        print("not wrapper")
        return gym.make(env_name)
        
    else:
        print("wrapper")
        env = make_atari(env_name)
        env = wrap_deepmind(env, frame_stack=True, scale=True)
        return env

