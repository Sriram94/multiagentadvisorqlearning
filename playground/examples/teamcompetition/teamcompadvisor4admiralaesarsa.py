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
        agents.Advisor4admiralaeteamcomp(372, sess),
        agents.DeepsarsaAgent(372, sess),
        agents.Advisor4admiralaeteamcomp(372, sess),
    ]
    env = pommerman.make('PommeTeam-v0', agent_list)

    env.seed(1)

    sess.run(tf.global_variables_initializer())
    
    with open('pommermanteamcompetitionexpert4offpolicy.csv', 'w+') as myfile:
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
            
            if agent_list[0].is_alive:
                actions_ = agent_list[1].act2(state_new[0], env.action_space)
            else:
                actions_ = -1


            if agent_list[1].is_alive:
                actions1_ = agent_list[1].act2(state_new[1], env.action_space)
            else:
                action1_ = -1


            if agent_list[2].is_alive:
                actions2_ = agent_list[1].act2(state_new[2], env.action_space)
            else:
                action2_ = -1



            if agent_list[3].is_alive:
                actions3_ = agent_list[1].act2(state_new[3], env.action_space)
            else:
                action3_ = -1



            action_list_next = [] 
            action_list_next.append(actions_)
            action_list_next.append(actions2_)
            action_list_next.append(actions3_)

            action_list_current = [] 
            action_list_current.append(actions[0])
            action_list_current.append(actions[2])
            action_list_current.append(actions[3])

            if agent_list[0].is_alive:
                actions0_new = agent_list[3].act2(state_new[0], env.action_space)
            else:
                actions0_new = -1

            if agent_list[1].is_alive:
                actions1_new = agent_list[3].act2(state_new[1], env.action_space)
            else:
                actions1_new = -1


            if agent_list[2].is_alive:
                actions2_new = agent_list[3].act2(state_new[2], env.action_space)
            else:
                actions2_new = -1

            if agent_list[3].is_alive:
                actions3_new = agent_list[3].act2(state_new[3], env.action_space)
            else:
                actions3_new = -1

            action_list_next_new = [] 
            action_list_next_new.append(actions0_new)
            action_list_next_new.append(actions1_new)
            action_list_next_new.append(actions2_new)

            action_list_current_new = [] 
            action_list_current_new.append(actions[0])
            action_list_current_new.append(actions[1])
            action_list_current_new.append(actions[2])
            
            agent_list[0].store(state[0], actions[0], reward[0], state_new[0], actions_new[0])
            agent_list[2].store(state[2], actions[2], reward[2], state_new[2], actions_new[2])

            agent_list[1].store(state[1], actions[1], action_list_current, reward[1], state_new[1], actions1_, action_list_next)
            agent_list[1].set(action_list_next)
            agent_list[3].store(state[3], actions[3], action_list_current_new, reward[3], state_new[3], actions3_new, action_list_next_new)
            agent_list[3].set(action_list_next_new)
            
            
            state = state_new
            actions = actions_new
        agent_list[0].learn()
        agent_list[1].learn()
        agent_list[2].learn()
        agent_list[3].learn()
        print("The rewards are", reward)
        cumulative_rewards[0] = cumulative_rewards[0] + reward[0]
        cumulative_rewards[1] = cumulative_rewards[1] + reward[1]
    
        with open('pommermanteamcompetitionexpert4offpolicy.csv', 'a') as myfile:
            myfile.write('{0},{1},{2}\n'.format(i_episode, cumulative_rewards[0], cumulative_rewards[1]))
        print('Episode {} finished'.format(i_episode))
    env.close()


if __name__ == '__main__':
    main()
