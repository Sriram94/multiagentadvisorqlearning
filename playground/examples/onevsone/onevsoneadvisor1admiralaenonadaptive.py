import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import pommerman
from pommerman import agents
import csv

def main():
    print(pommerman.REGISTRY)

    sess = tf.Session()

    agent_list = [
        agents.DQNAgent(201, sess),
        agents.Advisor1admiralaenonadaptive(201, sess),
    ]
    env = pommerman.make('OneVsOne-v0', agent_list)
    env.seed(1)


    sess.run(tf.global_variables_initializer())
    
    with open('pommermanonevsoneexpert1offpolicynew.csv', 'w+') as myfile:
        myfile.write('{0},{1},{2}\n'.format("Episode", "Reward1(DQN)","Reward2(ExpertQ)"))

    cumulative_rewards = []
    cumulative_rewards.append(0)
    cumulative_rewards.append(0)
    for i_episode in range(100000):
        state = env.reset()
        done = False
        actions = env.act(state)    
        while not done:
            state_new, reward, done, info = env.step(actions)
            actions_new = env.act(state_new)    
            actions_ = agent_list[1].act2(state_new[0], env.action_space)
            actions2_ = agent_list[1].act2(state_new[1], env.action_space)
            agent_list[0].store(state[0], actions[0], reward[0], state_new[0])
            agent_list[1].store(state[1], actions[1], actions[0], reward[1], state_new[1], actions_, actions2_)
            agent_list[1].set(actions_)
            state = state_new
            actions = actions_new
        agent_list[0].learn()
        agent_list[1].learn()
        print("The rewards are", reward)
        cumulative_rewards[0] = cumulative_rewards[0] + reward[0]
        cumulative_rewards[1] = cumulative_rewards[1] + reward[1]
    
        with open('pommermanonevsoneexpert1offpolicynew.csv', 'a') as myfile:
            myfile.write('{0},{1},{2}\n'.format(i_episode, cumulative_rewards[0], cumulative_rewards[1]))
        print('Episode {} finished'.format(i_episode))
    env.close()


if __name__ == '__main__':
    main()
