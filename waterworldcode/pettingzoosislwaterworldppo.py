from pettingzoo.sisl.waterworld import waterworld
from ppo import PPO
import csv
import numpy as np 

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

np.random.seed(1)



BATCH = 32
GAMMA = 0.9



def run_waterworld():
    
    step = 0
    with open('pettingzoosislwaterworldPPO.csv', 'w+') as myfile:
        myfile.write('{0},{1}\n'.format("Episode", "sumofrewards(PPO)"))
    num_episode = 0 
    while num_episode < 1000:
        agent_num = 0
        env.reset()
        buffer_s, buffer_a, buffer_r = [], [], []
        step = 0
        accumulated_reward = 0
        for agent in env.agent_iter():
            observation, reward, done, info = env.last()
            accumulated_reward = accumulated_reward + reward
            action = ppo.choose_action(observation)
            if done == False:
                env.step(action)
            step += 1
            buffer_s.append(observation)
            buffer_a.append(action)
            buffer_r.append(reward)
            agent_num = agent_num + 1
            if agent_num == len(env.agents):
                agent_num = 0
            
            if (step+1) % BATCH == 0 or done:

                v_s_ = ppo.get_v(observation)
                discounted_r = []
                for r in buffer_r[::-1]:
                    v_s_ = r + GAMMA * v_s_
                    discounted_r.append(v_s_)
                discounted_r.reverse()

                bs, ba, br = np.vstack(buffer_s), np.vstack(buffer_a), np.array(discounted_r)[:, np.newaxis]
                buffer_s, buffer_a, buffer_r = [], [], []
                ppo.update(bs, ba, br)
            
            if done:
                break
        
        with open('pettingzoosislwaterworldPPO.csv', 'a') as myfile:
            accumulated_reward = accumulated_reward/len(env.agents)
            myfile.write('{0},{1}\n'.format(num_episode, accumulated_reward))
        num_episode = num_episode + 1            
        print("We are in episode", num_episode)
    ppo.save_actor_model("./tmp/actor/ppomodel.ckpt")
    ppo.save_critic_model("./tmp/critic/ppomodel.ckpt")

    print('game over')


    #Code for loop that does both training and execution 
    #while num_episode < 1100:
    #    agent_num = 0
    #    env.reset()
    #    buffer_s, buffer_a, buffer_r = [], [], []
    #    step = 0
    #    accumulated_reward = 0
    #    for agent in env.agent_iter():
    #        observation, reward, done, info = env.last()
    #        accumulated_reward = accumulated_reward + reward
    #        action = ppo.choose_action(observation)
    #        if done == False:
    #            env.step(action)
    #        step += 1
    #        if num_episode < 1000:
    #            buffer_s.append(observation)
    #            buffer_a.append(action)
    #            buffer_r.append(reward)
    #        
    #        agent_num = agent_num + 1
    #        if agent_num == len(env.agents):
    #            agent_num = 0
    #        
    #        if num_episode < 1000:
    #            if (step+1) % BATCH == 0 or done:

    #                v_s_ = ppo.get_v(observation)
    #                discounted_r = []
    #                for r in buffer_r[::-1]:
    #                    v_s_ = r + GAMMA * v_s_
    #                    discounted_r.append(v_s_)
    #                discounted_r.reverse()

    #                bs, ba, br = np.vstack(buffer_s), np.vstack(buffer_a), np.array(discounted_r)[:, np.newaxis]
    #                buffer_s, buffer_a, buffer_r = [], [], []
    #                ppo.update(bs, ba, br)
    #        
    #        if done:
    #            break
    #    
    #    with open('pettingzoosislwaterworldPPO.csv', 'a') as myfile:
    #        accumulated_reward = accumulated_reward/len(env.agents)
    #        myfile.write('{0},{1}\n'.format(num_episode, accumulated_reward))
    #    num_episode = num_episode + 1            
    #    print("We are in episode", num_episode)
    #ppo.save_actor_model("./tmp/actor/ppomodel.ckpt")
    #ppo.save_critic_model("./tmp/critic/ppomodel.ckpt")

    #print('game over')



if __name__ == "__main__":
    env = waterworld.env()
    env.seed(1)
    sess = tf.Session()
    ppo = PPO(sess)
    sess.run(tf.global_variables_initializer())
    run_waterworld()


