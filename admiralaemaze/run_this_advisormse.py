from maze_env import Maze
from RL_brain_admiralae_mse import QLearningTable
import csv


with open('mse.csv', 'w+') as myfile2:
    myfile2.write('{0},{1}\n'.format("Episode", "mse"))


def update():
    cumulative_reward1 = 0
    cumulative_reward2 = 0

    for episode in range(2000):
        observation, observation2 = env.reset()
        action_tmp = 0
        action2_tmp = 0

        while True:
            env.render()

            observationcopy = observation.copy()
            observation2copy = observation2.copy()
            
            
            action, action2 = RL.choose_action(observationcopy, observation2copy, action_tmp, action2_tmp)
            
            observation_, observation2_, reward, reward2, done = env.step(action, action2)
            
            if observation_ != 'terminal': 
                observation_copy = observation_.copy()
            else: 
                observation_copy = observation_

            if observation2_ != 'terminal': 
                observation2_copy = observation2_.copy()
            else: 
                observation2_copy = observation2_

            observationcopy = observation.copy()
            observation2copy = observation2.copy()
            RL.learn(observationcopy, observation2copy, action, action2, reward, reward2, observation_copy, observation2_copy)
            
            action_tmp = action
            action2_tmp = action2
            observation = observation_
            observation2 = observation2_
            
            if done:
                break


        
        mse = RL.mean_square_error()
        
        with open('mse.csv', 'a') as myfile2:
            myfile2.write('{0},{1}\n'.format(episode, mse))
        print('Episode {} finished'.format(episode))

    print('game over')
    env.destroy()

if __name__ == "__main__":
    env = Maze()
    RL = QLearningTable(actions=list(range(env.n_actions)), actions2 = list(range(env.n_actions)))
    env.after(2000, update)
    env.mainloop()
