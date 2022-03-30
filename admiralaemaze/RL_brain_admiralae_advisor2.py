import numpy as np
import pandas as pd

UNIT = 40
MAZE_H = 5
MAZE_W = 5

class QLearningTable:
    def __init__(self, actions, actions2, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9):
        self.actions = actions  
        self.actions2 = actions2  
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
        if np.random.uniform() <= 0.5:
            action = self.advisor_action(observation)
            action2 = self.advisor_action(observation2)
        else:
            if np.random.uniform() < self.epsilon:
                state_action = self.q_table.loc[strobservation, :]
                state_action2 = self.q_table2.loc[strobservation2, :]
                action = np.random.choice(state_action[state_action == np.max(state_action)].index)
                action2 = np.random.choice(state_action2[state_action2 == np.max(state_action2)].index)
            else:
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
            advisoraction2 = self.advisor_action(s2_)
            advisoraction = self.advisor_action(s_)
            s_.append(advisoraction2)
            strs_ = str(s_)
            self.check_state_exist(strs_)
            q_target = r + self.gamma * self.q_table.loc[strs_, advisoraction]  
        else:
            q_target = r  
	
        self.q_table.loc[strs, a] += self.lr * (q_target - q_predict)  

        if s2_ != 'terminal':
            advisoraction = self.advisor_action(s_)
            advisoraction2 = self.advisor_action(s2_)
            s2_.append(advisoraction)
            strs2_ = str(s2_)
            self.check_state2_exist(strs2_)
            q_target2 = r2 + self.gamma * self.q_table2.loc[strs2_, advisoraction2]  
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
         
        if s[1] > UNIT * (MAZE_H - 1): 
            action = 0
        elif s[0] > UNIT * (MAZE_W - 1): 
            action = 3 
        elif s[0] < UNIT: 
            action = 2
        elif s[1] < UNIT: 
            action = 1 
        
        
        if s[1] > UNIT * 2 and s[0] > UNIT and s[1] < UNIT * 3 and s[0] < UNIT * 2: 
            action = 2
        
        elif s[1] > UNIT * 2 and s[0] > UNIT * 3 and s[1] < UNIT * 3 and s[0] < UNIT * 4: 
            action = 3

        if s[1] < UNIT and s[0] > UNIT * 2 and s[0] < UNIT * 3: 
            action = 2
        
        elif s[1] > UNIT * 4 and s[0] > UNIT * 2 and s[0] < UNIT * 3: 
            action = 2 

        
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
