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
        self.q_table3 = pd.DataFrame(columns=self.actions, dtype=np.float64)
        self.q_table4 = pd.DataFrame(columns=self.actions2, dtype=np.float64)

        self.read_csv()



    def choose_action(self, observation, observation2, action, action2): 
        observation_tmp = observation.copy()
        observation2_tmp = observation2.copy()
        observation.append(action2)
        observation.extend(observation2_tmp)
        strobservation = str(observation)
        observation2.append(action)
        observation2.extend(observation_tmp)
        strobservation2 = str(observation2)
        self.check_state_exist(strobservation)
        self.check_state2_exist(strobservation2)
        if np.random.uniform() <= 0.5: 
            action = self.advisor_action(observation_tmp)
            action2 = self.advisor_action(observation2_tmp)
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
        stmp = s.copy() 
        s2tmp = s2.copy() 
        s.append(a2)
        s.extend(s2tmp)
        s2.append(a)
        s2.extend(stmp)
        strs = str(s)
        strs2 = str(s2)
        
        self.check_state_exist(strs)
        self.check_state2_exist(strs2)
        q_predict = self.q_table.loc[strs, a]
        q_predict2 = self.q_table2.loc[strs2, a2]
        if s_ != 'terminal':
            s_tmp = s_.copy()
            advisoraction2 = self.advisor_action(s2_)
            advisoraction = self.advisor_action(s_tmp)
            s_.append(advisoraction2)
            s_.extend(s2_)
            strs_ = str(s_)
            
            self.check_state_exist(strs_)

            q_target = r + self.gamma * self.q_table.loc[strs_, advisoraction]  
        else:
            q_target = r  
	
        self.q_table.loc[strs, a] += self.lr * (q_target - q_predict)  
        
        
        if s2_ != 'terminal':
            advisoraction = self.advisor_action(s_tmp)
            advisoraction2 = self.advisor_action(s2_)
            s2_.append(advisoraction)
            s2_.extend(s_tmp)
            strs2_ = str(s2_)
            self.check_state2_exist(strs2_)
            q_target2 = r2 + self.gamma * self.q_table2.loc[strs2_, advisoraction2]  
        else:
            q_target2 = r2  
        
        self.q_table2.loc[strs2, a2] += self.lr * (q_target2 - q_predict2)  
        
        



    def mean_square_error(self): 
        mse = 0
        for index, row in self.q_table3.iterrows():
            if row[0] in self.q_table.index:
                error = self.q_table.loc[row[0],0] - row[1] 
                error = error * error 
                mse = mse + error 
                error = self.q_table.loc[row[0],1] - row[2] 
                error = error * error 
                mse = mse + error 
                error = self.q_table.loc[row[0],2] - row[3] 
                error = error * error 
                mse = mse + error 
                error = self.q_table.loc[row[0],3] - row[4] 
                error = error * error 
                mse = mse + error 
        size = self.q_table.size * 4
        mse = mse/size
        print("the mse is", mse)
        return mse


    def write_to_csv(self):
        
        self.q_table.to_csv('q_table1.csv')
        self.q_table2.to_csv('q_table2.csv')
        print("Files written")

    def read_csv(self): 
        self.q_table3 = pd.read_csv('q_table1.csv')
        self.q_table4 = pd.read_csv('q_table2.csv')
        print("Read both the files")

    def advisor_action(self,s):
         
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

        elif s[1] > UNIT and s[1] < UNIT * 2 and s[0] > UNIT and s[0] < UNIT * 2: 
            action = 1 

        elif s[1] > UNIT * 3 and s[1] < UNIT * 4 and s[0] > UNIT and s[0] < UNIT * 2: 
            action = 0 

        elif s[1] > UNIT and s[1] < UNIT * 2 and s[0] > UNIT * 3 and s[0] < UNIT * 4: 
            action = 1 
        
        elif s[1] > UNIT * 3 and s[1] < UNIT * 4 and s[0] > UNIT * 3 and s[0] < UNIT * 4: 
            action = 0 

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

