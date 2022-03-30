import numpy as np
import pandas as pd

UNIT = 40
MAZE_H = 5
MAZE_W = 5

class QLearningTable:
    def __init__(self, actions, actions2, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9):
        self.actions = actions  # a list
        self.actions2 = actions2  # a list
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy
        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)
        self.q_table2 = pd.DataFrame(columns=self.actions2, dtype=np.float64)

    def choose_action(self, observation, observation2, action, action2): 
        observation.append(action2)
        observation2.append(action)
        strobservation = str(observation)
        strobservation2 = str(observation2)
        self.check_state_exist(strobservation)
        self.check_state2_exist(strobservation2)
        # action selection
        if np.random.uniform() < self.epsilon:
            # choose best action
            state_action = self.q_table.loc[strobservation, :]
            state_action2 = self.q_table2.loc[strobservation2, :]
            # some actions may have the same value, randomly choose on in these actions
            action = np.random.choice(state_action[state_action == np.max(state_action)].index)
            action2 = np.random.choice(state_action2[state_action2 == np.max(state_action2)].index)
        else:
            # choose random action
            action = np.random.choice(self.actions)
            action2 = np.random.choice(self.actions2)
        return action, action2

    def learn(self, s, s2, a, a2, r, r2, s_, s2_):
        s.append(a2)
        s2.append(a)
        strs = str(s)
        strs2 = str(s2)
        self.check_state_exist(strs)
        self.check_state2_exist(strs2)
        q_predict = self.q_table.loc[strs, a]
        q_predict2 = self.q_table2.loc[strs2, a2]
        if s_ != 'terminal':
            s_.append(a2)
            strs_ = str(s_)
            self.check_state_exist(strs_)
            q_target = r + self.gamma * self.q_table.loc[strs_, :].max()  # next state is not terminal
        else:
            q_target = r  # next state is terminal
	
        self.q_table.loc[strs, a] += self.lr * (q_target - q_predict)  # update

        if s2_ != 'terminal':
            s2_.append(a)
            strs2_ = str(s2_)
            self.check_state2_exist(strs2_)
            q_target2 = r2 + self.gamma * self.q_table2.loc[strs2_, :].max()  # next state is not terminal
        else:
            q_target2 = r2  # next state is terminal
        
        self.q_table2.loc[strs2, a2] += self.lr * (q_target2 - q_predict2)  # update


    def check_state_exist(self, state):
        if state not in self.q_table.index:
            # append new state to q table
            self.q_table = self.q_table.append(
                pd.Series(
                    [0]*len(self.actions),
                    index=self.q_table.columns,
                    name=state,
                )
            )


    def check_state2_exist(self, state):
        if state not in self.q_table2.index:
            # append new state to q table
            self.q_table2 = self.q_table2.append(
                pd.Series(
                    [0]*len(self.actions2),
                    index=self.q_table2.columns,
                    name=state,
                )
            )
