from pettingzoo.sisl.waterworld import waterworld
from ddpg import Actor
from ddpg import Critic
from ddpg import Memory
import csv
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

np.random.seed(1)

MAX_EPISODES = 200
MAX_EP_STEPS = 200
LR_A = 0.001    
LR_C = 0.002    
GAMMA = 0.9     
REPLACEMENT = [
    dict(name='soft', tau=0.01),
    dict(name='hard', rep_iter_a=600, rep_iter_c=500)
][0]            
MEMORY_CAPACITY = 10000
BATCH_SIZE = 32

state_dim = 212
action_dim = 2
action_bound = 1


def run_waterworld():
    
    step = 0
    with open('pettingzoosislwaterworldDDPG.csv', 'w+') as myfile:
        myfile.write('{0},{1}\n'.format("Episode", "sumofrewards(DDPG)"))
    num_episode = 0 
    var = 3 
    while num_episode < 1000:
        agent_num = 0
        env.reset()
        step = 0
        observation_tmp = []
        
        obs_list = [[] for _ in range(len(env.agents))]
        action_list = [[] for _ in range(len(env.agents))]
        reward_list = [[] for _ in range(len(env.agents))]
        
        accumulated_reward = 0
        
        for agent in env.agent_iter():
            observation, reward, done, info = env.last()
            accumulated_reward = accumulated_reward + reward
            action = actor.choose_action(observation)
            action = np.clip(np.random.normal(action, var), -action_bound, action_bound) 
            obs_list[agent_num].append(observation)
            action_list[agent_num].append(action)
            reward_list[agent_num].append(reward)
            if len(obs_list[agent_num]) == 2:
                M.store_transition(obs_list[agent_num][0], action_list[agent_num][0], reward_list[agent_num][0], obs_list[agent_num][1])
                obs_list[agent_num].pop(0)
                action_list[agent_num].pop(0)
                reward_list[agent_num].pop(0)
            
            if M.pointer > MEMORY_CAPACITY:
                var *= .9995    
                b_M = M.sample(BATCH_SIZE)
                b_s = b_M[:, :state_dim]
                b_a = b_M[:, state_dim: state_dim + action_dim]
                b_r = b_M[:, -state_dim - 1: -state_dim]
                b_s_ = b_M[:, -state_dim:]

                critic.learn(b_s, b_a, b_r, b_s_)
                actor.learn(b_s)
            
            if done == False:
                env.step(action)
            step += 1
            agent_num = agent_num + 1
            if agent_num == len(env.agents):
                agent_num = 0
             
            if done:
                break
        
        with open('pettingzoosislwaterworldDDPG.csv', 'a') as myfile:
            accumulated_reward = accumulated_reward/len(env.agents)
            myfile.write('{0},{1}\n'.format(num_episode, accumulated_reward))
        num_episode = num_episode + 1            
        print("We are in episode", num_episode) 
    print('game over')



    #Code for loop that does both training and execution 
    #while num_episode < 1100:
    #    agent_num = 0
    #    env.reset()
    #    step = 0
    #    observation_tmp = []
    #    
    #    obs_list = [[] for _ in range(len(env.agents))]
    #    action_list = [[] for _ in range(len(env.agents))]
    #    reward_list = [[] for _ in range(len(env.agents))]
    #    
    #    accumulated_reward = 0
    #    
    #    for agent in env.agent_iter():
    #        observation, reward, done, info = env.last()
    #        accumulated_reward = accumulated_reward + reward
    #        action = actor.choose_action(observation)
    #        if num_episode < 1000:
    #            action = np.clip(np.random.normal(action, var), -action_bound, action_bound) 
    #        else:
    #            action = np.clip(action, -action_bound, action_bound) 

    #        if num_episode < 1000:
    #            obs_list[agent_num].append(observation)
    #            action_list[agent_num].append(action)
    #            reward_list[agent_num].append(reward)
    #            if len(obs_list[agent_num]) == 2:
    #                M.store_transition(obs_list[agent_num][0], action_list[agent_num][0], reward_list[agent_num][0], obs_list[agent_num][1])
    #                obs_list[agent_num].pop(0)
    #                action_list[agent_num].pop(0)
    #                reward_list[agent_num].pop(0)
    #        
    #            if M.pointer > MEMORY_CAPACITY:
    #                var *= .9995    
    #                b_M = M.sample(BATCH_SIZE)
    #                b_s = b_M[:, :state_dim]
    #                b_a = b_M[:, state_dim: state_dim + action_dim]
    #                b_r = b_M[:, -state_dim - 1: -state_dim]
    #                b_s_ = b_M[:, -state_dim:]

    #                critic.learn(b_s, b_a, b_r, b_s_)
    #                actor.learn(b_s)
    #        
    #        if done == False:
    #            env.step(action)
    #        step += 1
    #        agent_num = agent_num + 1
    #        if agent_num == len(env.agents):
    #            agent_num = 0
    #         
    #        if done:
    #            break
    #    
    #    with open('pettingzoosislwaterworldDDPG.csv', 'a') as myfile:
    #        accumulated_reward = accumulated_reward/len(env.agents)
    #        myfile.write('{0},{1}\n'.format(num_episode, accumulated_reward))
    #    num_episode = num_episode + 1            
    #    print("We are in episode", num_episode) 
    #print('game over')



if __name__ == "__main__":
    env = waterworld.env()
    env.seed(1)
    sess = tf.Session()
    actor = Actor(sess, action_dim, action_bound, LR_A, REPLACEMENT)
    critic = Critic(sess, state_dim, action_dim, LR_C, GAMMA, REPLACEMENT, actor.a, actor.a_)
    actor.add_grad_to_graph(critic.a_grads)
    sess.run(tf.global_variables_initializer())
    M = Memory(MEMORY_CAPACITY, dims=2 * state_dim + action_dim + 1)
    run_waterworld()

