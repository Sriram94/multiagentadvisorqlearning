from . import BaseAgent
from RL_brain_admiralae import AdmiralaeNetwork
from .. import characters
import os
import sys
import numpy as np



from collections import defaultdict
import queue
import random


from .. import constants
from .. import utility


np.random.seed(1)
random.seed(1)






def make_np_float(feature):
    return np.array(feature).astype(np.float32)







def featurize(obs):
    board = obs["board"].reshape(-1).astype(np.float32)
    bomb_blast_strength = obs["bomb_blast_strength"].reshape(-1).astype(np.float32)
    bomb_life = obs["bomb_life"].reshape(-1).astype(np.float32)
    position = make_np_float(obs["position"])
    ammo = make_np_float([obs["ammo"]])
    blast_strength = make_np_float([obs["blast_strength"]])
    can_kick = make_np_float([obs["can_kick"]])

    teammate = obs["teammate"]
    if teammate is not None:
        teammate = teammate.value
    else:
        teammate = -1
    teammate = make_np_float([teammate])

    enemies = obs["enemies"]
    enemies = [e.value for e in enemies]
    if len(enemies) < 3:
        enemies = enemies + [-1]*(3 - len(enemies))
    enemies = make_np_float(enemies)

    return np.concatenate((board, bomb_blast_strength, bomb_life, position, ammo, blast_strength, can_kick, teammate, enemies))





class Advisor4admiralae(BaseAgent):
    def __init__(self, nf, sess, *args, **kwargs):
        super(Advisor4admiralae, self).__init__(*args, **kwargs)
        self._recently_visited_positions = []
        self._recently_visited_length = 6
        self._prev_direction = None    
        self.counter = 0
        self.epsilon = 0.5
        self.RL = AdmiralaeNetwork(sess, 6, nf,
                      learning_rate=0.01,
                      reward_decay=0.9,
                      e_greedy=0.9,
                      memory_size=2000000)

        self.action2 = 0
        self.action_space = 6


    def set(self, action2):
        self.action2 = action2

    def act(self, obs, action_space):
        if (np.random.uniform() <= 0.5):
            return self.act2(obs, action_space)
        
        else:
            obs = featurize(obs)
            return self.RL.choose_action(obs,self.action2)


    def store(self, obs, act, act_other, reward, obs_, act_, act_other_):
        obs = featurize(obs)
        obs_ = featurize(obs_)
        self.RL.store_transition(obs, act, act_other, reward, obs_, act_, act_other_)

    
    def learn(self):
        self.RL.learn()

    def copy_network(self, s):
        self.RL.copy_network(s)

    def save_model(self,s):
        self.RL.save_model(s)
    
    def act2(self, obs, action_space):
        return action_space.sample()









    







