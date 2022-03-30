from pettingzoo.sisl.pursuit import pursuit
from RL_brain_DQN import DeepQNetwork
from RL_brain_CHAT import CHAT
import csv
import numpy as np 

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

np.random.seed(1)

def change_observation(observation):
    observation = observation.tolist()
    new_list = []
    for i in range(len(observation)):
        for j in range(len(observation[i])):
            for k in range(len(observation[i][j])):
                new_list.append(observation[i][j][k])
    new_observation = np.array(new_list)
    return new_observation



def run_pursuit():
    
    step = 0
    with open('pettingzoosislpursuitCHAT.csv', 'w+') as myfile:
        myfile.write('{0},{1}\n'.format("Episode", "sumofrewards(CHAT)"))
    
    num_episode = 0 
    while num_episode < 1000:
        agent_num = 0
        env.reset()
        obs_list = [[] for _ in range(len(env.agents))]
        action_list = [[] for _ in range(len(env.agents))]
        reward_list = [[] for _ in range(len(env.agents))]
        advisoraction_list = [[] for _ in range(len(env.agents))]
        accumulated_reward = 0
        for agent in env.agent_iter():
            observation, reward, done, info = env.last()
            observation = change_observation(observation)
            accumulated_reward = accumulated_reward + reward
            obs_list[agent_num].append(observation)
            action = RL2.choose_action(observation)
            action2 = RL.choose_action(observation, execution = True)
            if action == 5:
                action = action2
                conf = RL2.get_confidence(observation, action)
                if conf<0.6:
                    action = 0
            action_list[agent_num].append(action)
            advisoraction_list[agent_num].append(action2)
            reward_list[agent_num].append(reward)
            if len(obs_list[agent_num]) == 2:
                RL2.store_transition(obs_list[agent_num][0], action_list[agent_num][0], reward_list[agent_num][0], obs_list[agent_num][1], action_list[agent_num][1], advisoraction_list[agent_num][0])
                obs_list[agent_num].pop(0)
                action_list[agent_num].pop(0)
                reward_list[agent_num].pop(0)
            if done == False:
                env.step(action)
            step += 1
            if (step > 200) and (step % 5 == 0):
                RL2.learn()
            agent_num = agent_num + 1
            if agent_num == len(env.agents):
                agent_num = 0
            if done:
                break
            
        with open('pettingzoosislpursuitCHAT.csv', 'a') as myfile:
            accumulated_reward = accumulated_reward/len(env.agents)
            myfile.write('{0},{1}\n'.format(num_episode, accumulated_reward))
        num_episode = num_episode + 1            
        print("We are in episode", num_episode)
    print('game over')



    # Code for loop that does both training and execution
    #while num_episode < 1100:
    #    agent_num = 0
    #    env.reset()
    #    obs_list = [[] for _ in range(len(env.agents))]
    #    action_list = [[] for _ in range(len(env.agents))]
    #    reward_list = [[] for _ in range(len(env.agents))]
    #    advisoraction_list = [[] for _ in range(len(env.agents))]
    #    accumulated_reward = 0
    #    for agent in env.agent_iter():
    #        observation, reward, done, info = env.last()
    #        observation = change_observation(observation)
    #        accumulated_reward = accumulated_reward + reward
    #        if num_episode >= 1000: 
    #            action = RL2.choose_action(observation, execution=True)
    #        else: 
    #            action = RL2.choose_action(observation)

    #        action2 = RL.choose_action(observation, execution=True)
    #        if action == 5:
    #            action = action2
    #            conf = RL2.get_confidence(observation, action)
    #            if conf<0.6:
    #                action = 0
    #        
    #        if num_episode < 1000:
    #            obs_list[agent_num].append(observation)
    #            action_list[agent_num].append(action)
    #            advisoraction_list[agent_num].append(action2)
    #            reward_list[agent_num].append(reward)
    #            if len(obs_list[agent_num]) == 2:
    #                RL2.store_transition(obs_list[agent_num][0], action_list[agent_num][0], reward_list[agent_num][0], obs_list[agent_num][1], action_list[agent_num][1], advisoraction_list[agent_num][0])
    #                obs_list[agent_num].pop(0)
    #                action_list[agent_num].pop(0)
    #                reward_list[agent_num].pop(0)
    #        
    #        
    #        if done == False:
    #            env.step(action)
    #        step += 1
    #        
    #        
    #        if num_episode < 1000: 
    #            if (step > 200) and (step % 5 == 0):
    #                RL2.learn()
    #        
    #        agent_num = agent_num + 1
    #        if agent_num == len(env.agents):
    #            agent_num = 0
    #        if done:
    #            break
    #        
    #    with open('pettingzoosislpursuitCHAT.csv', 'a') as myfile:
    #        accumulated_reward = accumulated_reward/len(env.agents)
    #        myfile.write('{0},{1}\n'.format(num_episode, accumulated_reward))
    #    num_episode = num_episode + 1            
    #    print("We are in episode", num_episode)
    #print('game over')





if __name__ == "__main__":
    env = pursuit.env()
    env.seed(1)
    sess = tf.Session()
    RL = DeepQNetwork(sess, 5,147,
                      learning_rate=0.01,
                      reward_decay=0.9,
                      e_greedy=1,
                      memory_size=2000000,
                      )
    RL.restore_model("./tmp/dqnmodel.ckpt")
    
    RL2 = CHAT(sess, 5, 147, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9, memory_size=2000000)


    sess.run(tf.global_variables_initializer())
    run_pursuit()


