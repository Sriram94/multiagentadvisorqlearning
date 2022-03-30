import numpy as np
import pandas as pd

UNIT = 40
MAZE_H = 5
MAZE_W = 5

class QLearningTable:
    def __init__(self, actions, actions2, learning_rate=0.01, reward_decay=0.9, e_greedy=0.1):
        self.actions = actions  
        self.actions2 = actions2  
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy
        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)
        self.q_table2 = pd.DataFrame(columns=self.actions2, dtype=np.float64)

    def linear_decay(self, epoch, x, y):
        min_v, max_v = y[0], y[-1]
        start, end = x[0], x[-1]

        if epoch == start:
            return min_v

        eps = min_v

        for i, x_i in enumerate(x):
            if epoch <= x_i:
                interval = (y[i] - y[i - 1]) / (x_i - x[i - 1])
                eps = interval * (epoch - x[i - 1]) + y[i - 1]
                break

        return eps




    def choose_action(self, observation, observation2, action, action2, episode): 
        
        if observation == 'terminal' or observation2 == 'terminal':
            action = 0
            action2 = 0
            return action, action2

        else:

            observation.append(action2)
            observation2.append(action)
            strobservation = str(observation)
            strobservation2 = str(observation2)
            self.check_state_exist(strobservation)
            self.check_state2_exist(strobservation2)
            k = 0
            t = self.epsilon

            if np.random.uniform() <= k: 
                action = self.advisor_action(observation)
                action2 = self.advisor_action(observation2)
            
            else:
                if np.random.uniform() > t:
                    state_action = self.q_table.loc[strobservation, :]
                    state_action2 = self.q_table2.loc[strobservation2, :]
                    action = np.random.choice(state_action[state_action == np.max(state_action)].index)
                    action2 = np.random.choice(state_action2[state_action2 == np.max(state_action2)].index)
                else:
                    action = np.random.choice(self.actions)
                    action2 = np.random.choice(self.actions2)
            return action, action2

    def choose_greedy_action(self, observation, observation2, action, action2):         
    
        if observation == 'terminal' or observation2 == 'terminal':
            action = 0
            action2 = 0
            return action, action2
        
        else:
        
            observation.append(action2)
            observation2.append(action)
            strobservation = str(observation)
            strobservation2 = str(observation2)
            self.check_state_exist(strobservation)
            self.check_state2_exist(strobservation2)
            state_action = self.q_table.loc[strobservation, :]
            state_action2 = self.q_table2.loc[strobservation2, :]
            action = np.random.choice(state_action[state_action == np.max(state_action)].index)
            action2 = np.random.choice(state_action2[state_action2 == np.max(state_action2)].index)
            return action, action2




    def learn(self, s, s2, a, a2, r, r2, s_, s2_, a_, a2_):
        s.append(a2)
        s2.append(a)
        strs = str(s)
        strs2 = str(s2)
        self.check_state_exist(strs)
        self.check_state2_exist(strs2)
        q_predict = self.q_table.loc[strs, a]
        q_predict2 = self.q_table2.loc[strs2, a2]
        if s_ != 'terminal':
            s_.append(a2_)
            strs_ = str(s_)
            self.check_state_exist(strs_)
            q_target = r + self.gamma * self.q_table.loc[strs_, a_]  
        else:
            q_target = r  
	
        self.q_table.loc[strs, a] += self.lr * (q_target - q_predict)  

        if s2_ != 'terminal':
            s2_.append(a_)
            strs2_ = str(s2_)
            self.check_state2_exist(strs2_)
            q_target2 = r2 + self.gamma * self.q_table2.loc[strs2_, a2_]  
        else:
            q_target2 = r2  
        
        self.q_table2.loc[strs2, a2] += self.lr * (q_target2 - q_predict2)  
        #if s_ == 'terminal' or s2_ == 'terminal':
        #    print("The q table for 1st agent is")
        #    print(self.q_table)  
        #    print("The q table for 2nd agent is")
        #    print(self.q_table2) 

    def advisor_action(self,s):

        action = np.random.choice(self.actions) 
        return action 


    def check_state_exist(self, state):
        if state not in self.q_table.index:
            self.q_table = self.q_table.append(
                pd.Series(
                    [0]*len(self.actions),
                    index=self.q_table.columns,
                    name=state,
                )
            )


    def check_state2_exist(self, state):
        if state not in self.q_table2.index:
            self.q_table2 = self.q_table2.append(
                pd.Series(
                    [0]*len(self.actions2),
                    index=self.q_table2.columns,
                    name=state,
                )
            )
