import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import pommerman
from pommerman import agents
import csv

def main():
    print(pommerman.REGISTRY)

    sess = tf.Session()

    agent_list = [
        agents.DeepsarsaAgent(372, sess),
        agents.Advisor3(372, False, sess),
        agents.DeepsarsaAgent(372, sess),
        agents.Advisor3(372, False, sess),
    ]    
    env = pommerman.make('PommeTeam-v0', agent_list)

    env.seed(1)

    sess.run(tf.global_variables_initializer())
    
    with open('pommermanteamcompetitionexpert3.csv', 'w+') as myfile:
        myfile.write('{0},{1},{2},{3},{4}\n'.format("Episode", "Reward1(DQN)", "Reward2(DQNExpert)", "Reward1(DQN)", "Reward1(DQNExpert)"))
                
    
    cumulative_rewards = []
    cumulative_rewards.append(0)
    cumulative_rewards.append(0)
    cumulative_rewards.append(0)
    cumulative_rewards.append(0)
    for i_episode in range(100000):
        state = env.reset()
        done = False
        actions = env.act(state)    
        while not done:
            state_new, reward, done, info = env.step(actions)
            actions_new = env.act(state_new)    
            agent_list[0].store(state[0], actions[0], reward[0], state_new[0], actions_new[0])
            agent_list[2].store(state[2], actions[2], reward[2], state_new[2], actions_new[2])
            action_list = []
            action_list.append(actions[0])
            action_list.append(actions[2])
            action_list.append(actions[3])
            
            action_list2 = []
            action_list2.append(actions[0])
            action_list2.append(actions[1])
            action_list2.append(actions[2])
            
            action_list_new = []
            action_list_new2 = []

            action_list_new.append(actions_new[0])
            action_list_new.append(actions_new[2])
            action_list_new.append(actions_new[3])
            action_list_new2.append(actions_new[0])
            action_list_new2.append(actions_new[1])
            action_list_new2.append(actions_new[2])
            agent_list[1].store(state[1], actions[1], action_list, action_list_new, reward[1], state_new[1], actions_new[1])
            agent_list[3].store(state[3], actions[3], action_list2, action_list_new2, reward[3], state_new[3], actions_new[3])
            agent_list[1].set(action_list_new)
            agent_list[3].set(action_list_new2)
            state = state_new
            actions = actions_new
        agent_list[0].learn()
        agent_list[1].learn()
        agent_list[2].learn()
        agent_list[3].learn()
        print("The rewards are", reward)
        cumulative_rewards[0] = cumulative_rewards[0] + reward[0]
        cumulative_rewards[1] = cumulative_rewards[1] + reward[1]
        cumulative_rewards[2] = cumulative_rewards[2] + reward[2]
        cumulative_rewards[3] = cumulative_rewards[3] + reward[3]
    
        with open('pommermanteamcompetitionexpert3.csv', 'a') as myfile:
            myfile.write('{0}, {1}, {2}, {3}, {4}\n'.format(i_episode, cumulative_rewards[0], cumulative_rewards[1], cumulative_rewards[2], cumulative_rewards[3]))
        print('Episode {} finished'.format(i_episode))
    env.close()


if __name__ == '__main__':
    main()
