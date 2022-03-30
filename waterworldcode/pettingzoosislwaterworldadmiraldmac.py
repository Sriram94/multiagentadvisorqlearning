from pettingzoo.sisl.waterworld import waterworld
from ppo import PPO
from RL_brain_admiraldmac import Actor
from RL_brain_admiraldmac import Critic
import csv
import numpy as np 

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

np.random.seed(1)


def linear_decay(epoch, x, y):
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






def run_waterworld():
    
    step = 0
    with open('pettingzoosislwaterworldadvisorac.csv', 'w+') as myfile:
        myfile.write('{0},{1}\n'.format("Episode", "sumofrewards(AC)")) 
    num_episode = 0 
    eps = 0.8 
    while num_episode < 1000:
        agent_num = 0
        env.reset()
        obs_list = [[] for _ in range(len(env.agents))]
        action_list = [[] for _ in range(len(env.agents))]
        reward_list = [[] for _ in range(len(env.agents))]
        accumulated_reward = 0
        for i in range(len(env.agents)):
            list_new = [0,0]
            action_list[i].append(list_new)
        for agent in env.agent_iter():
            observation, reward, done, info = env.last()
            accumulated_reward = accumulated_reward + reward
            obs_list[agent_num].append(observation)
            if np.random.uniform() <= eps:
                action = ppo.choose_action(observation)
            else:
                action = actor.choose_action(observation)
            curaction_list = action.tolist()
            action_list[agent_num].append(curaction_list)
            
            reward_list[agent_num].append(reward)
            

            if len(obs_list[agent_num]) == 2:
                
                action_opp = []
                for i in range(len(env.agents)):
                    if i != agent_num:
                        action_opp.append(action_list[i][0])
                
                action_opp_new = []
                for i in range(len(env.agents)):
                    if i != agent_num:
                        action_opp_new.append(action_list[i][1])
                td_error = critic.learn(obs_list[agent_num][0], action_opp, reward_list[agent_num][0], obs_list[agent_num][1], action_opp_new)
                actor.learn(obs_list[agent_num][0], action_list[agent_num][0], td_error)
            
            if len(obs_list[agent_num]) == 2:
                obs_list[agent_num].pop(0)
                action_list[agent_num].pop(0)
                reward_list[agent_num].pop(0)
            
            if done == False:
                env.step(action)
            
            step += 1
            
            
            agent_num = agent_num + 1
            
            if agent_num == len(env.agents):
                agent_num = 0
            
            if done:
                break
            
        with open('pettingzoosislwaterworldadvisorac.csv', 'a') as myfile:
            accumulated_reward = accumulated_reward/len(env.agents)
            myfile.write('{0},{1}\n'.format(num_episode, accumulated_reward))
        num_episode = num_episode + 1       
        print("we are in episode", num_episode)    
        eps = linear_decay(num_episode, [0, int(1000 * 0.99), 1000], [0.8, 0.2, 0.01])
    print('game over')


    #Code for loop that does both training and execution 
    #while num_episode < 1100:
    #    agent_num = 0
    #    env.reset()
    #    obs_list = [[] for _ in range(len(env.agents))]
    #    action_list = [[] for _ in range(len(env.agents))]
    #    reward_list = [[] for _ in range(len(env.agents))]
    #    accumulated_reward = 0
    #    for i in range(len(env.agents)):
    #        list_new = [0,0]
    #        action_list[i].append(list_new)
    #    for agent in env.agent_iter():
    #        observation, reward, done, info = env.last()
    #        accumulated_reward = accumulated_reward + reward
    #        if np.random.uniform() <= eps:
    #            action = ppo.choose_action(observation)
    #        else:
    #            action = actor.choose_action(observation)
    #        
    #        if num_episode < 1000: 
    #            curaction_list = action.tolist()
    #            obs_list[agent_num].append(observation)
    #            action_list[agent_num].append(curaction_list)
    #            
    #            reward_list[agent_num].append(reward)
    #            

    #            if len(obs_list[agent_num]) == 2:
    #                
    #                action_opp = []
    #                for i in range(len(env.agents)):
    #                    if i != agent_num:
    #                        action_opp.append(action_list[i][0])
    #                
    #                action_opp_new = []
    #                for i in range(len(env.agents)):
    #                    if i != agent_num:
    #                        action_opp_new.append(action_list[i][1])
    #                td_error = critic.learn(obs_list[agent_num][0], action_opp, reward_list[agent_num][0], obs_list[agent_num][1], action_opp_new)
    #                actor.learn(obs_list[agent_num][0], action_list[agent_num][0], td_error)
    #            
    #            if len(obs_list[agent_num]) == 2:
    #                obs_list[agent_num].pop(0)
    #                action_list[agent_num].pop(0)
    #                reward_list[agent_num].pop(0)
    #        
    #        if done == False:
    #            env.step(action)
    #        
    #        step += 1
    #        
    #        
    #        agent_num = agent_num + 1
    #        
    #        if agent_num == len(env.agents):
    #            agent_num = 0
    #        
    #        if done:
    #            break
    #        
    #    with open('pettingzoosislwaterworldadvisorac.csv', 'a') as myfile:
    #        accumulated_reward = accumulated_reward/len(env.agents)
    #        myfile.write('{0},{1}\n'.format(num_episode, accumulated_reward))
    #    num_episode = num_episode + 1       
    #    print("we are in episode", num_episode)    
    #    if num_episode < 1000:
    #        eps = linear_decay(num_episode, [0, int(1000 * 0.99), 1000], [0.8, 0.2, 0.01])
    #    else: 
    #        eps = 0 
    #print('game over')




if __name__ == "__main__":
    env = waterworld.env()
    env.seed(1)
    sess = tf.Session()
    ppo = PPO(sess)
    ppo.restore_actor_model("./tmp/actor/ppomodel.ckpt")
    ppo.restore_critic_model("./tmp/critic/ppomodel.ckpt")
 
    action_bound = [-1, 1]
    actor = Actor(sess, n_features=212,  action_bound = action_bound, lr=0.00001)
    critic = Critic(sess, n_features=212, lr=0.001)    

    sess.run(tf.global_variables_initializer())
    run_waterworld()


