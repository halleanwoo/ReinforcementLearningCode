import tensorflow as tf
import collections
import numpy as np
import random

# memory for momery replay
# memory = []
Transition = collections.namedtuple("Transition" , ["state", "action", "reward", "next_state", "done", "episode_return"])
class Memory:
    def __init__(self, size, flag_piexl=0):
        self.memory = []
        self.capacity = size
        self.flag_piexl = flag_piexl

    def store(self, state, action, reward, next_state, done, ep_return=0):
        if len(self.memory) > self.capacity:
            self.memory.pop(0)
        if self.flag_piexl:
            assert np.amin(state) >= 0.0
            assert np.amax(state) <= 1.0

            # Class LazyFrame --> np.array()
            state = np.array(state)
            next_state = np.array(next_state)

            state  = (state * 255).round().astype(np.uint8)
            next_state = (next_state * 255).round().astype(np.uint8)

        self.memory.append(Transition(state, action , reward , next_state , float(done), ep_return))
 
    def batchSample(self, batch_size):
        batch_transition = random.sample(self.memory, batch_size)
        state, action, reward, next_state, done, ep_return = map(np.array , zip(*batch_transition))
        if self.flag_piexl:
            state = state.astype(np.float32) / 255.0
            next_state = next_state.astype(np.float32) / 255.0
        return state, action, reward, next_state, done, ep_return

    def size(self):
        return len(self.memory)
