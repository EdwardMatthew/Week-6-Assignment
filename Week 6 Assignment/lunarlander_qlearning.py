import gym
import numpy as np
import matplotlib.pyplot as plt
import time

env = gym.make("LunarLander-v2")

# initializing Qtable
def Qtable(state_space, action_space, bin_size=10):
    bins = [np.linspace(-1.5, 1.5, bin_size),
            np.linspace(-1.5, 1.5, bin_size),
            np.linspace(-5, 5, bin_size),
            np.linspace(-5, 5, bin_size),
            np.linspace(-3.14, 3.14, bin_size),
            np.linspace(-5, 5, bin_size),
            np.linspace(-0, 1, bin_size),
            np.linspace(-0, 1, bin_size)]

    qtable = np.random.uniform(low=-1, high=1, size=([bin_size] * state_space + [action_space]))
    return qtable, bins

def Discrete(state, bins):
    index = []
    for i in range(len(state)):
        index.append(np.digitize(state[i], bins[i]) - 1)
    return tuple(index)

def q_learning(q_table, bins, episodes=5000, gamma=0.95, lr=0.1, timestep=5000, epsilon=0.2):
    rewards = 0
    solved = False
    steps = 0
    runs = [0]
    data = {'max' : [0], 'avg' : [0]}
    ep = [i for i in range(0,episodes + 1,timestep)]
    
    # iterating through episodes
    for episode in range(1, episodes+1):
        current_state = Discrete(env.reset()[0], bins) # initial observation
        score = 0
        terminated = False

        while not terminated:
            steps += 1
            ep_start = time.time()
            if episode%timestep==0:
                env.render()

            if np.random.uniform(0,1) < epsilon:
                # exploration
                action = env.action_space.sample()
            else:
                # if randomness below threshold, switch to exploitation
                action = np.argmax(q_table[current_state])

            observation, reward, terminated, truncated, info = env.step(action)

            next_state = Discrete(observation, bins)

            score += reward

            # reinforcement learning, update the q-value everytime agent fails
            if not terminated:
                max_future_q = np.max(q_table[next_state])
                current_q = q_table[current_state+(action,)]
                new_q = (1-lr)*current_q + lr*(reward + gamma*max_future_q)
                q_table[current_state+(action,)] = new_q

            # update state
            current_state = next_state
       
        # update timestep value
        else:
            rewards += score
            runs.append(score)
            if score > 200 and steps >= 100 and solved == False:
                solved = True
                print('Solved in episode : {} in time : {}'.format(episode, (time.time()-ep_start)))

        # Updating timestep value
        if episode%timestep == 0:
            print('Episode : {} | Reward -> {} | Max reward : {} | Time : {}'.format(episode, rewards/timestep, max(runs), time.time() - ep_start))
            data['max'].append(max(runs))
            data['avg'].append(rewards/timestep)
            if rewards/timestep >= 200:
                print('Solved in episode : {}'.format(episode))
            rewards, runs = 0,[0]

    # plot the results
    if len(ep) == len(data['max']):
        plt.plot(ep, data['max'], label='Max')
        plt.plot(ep, data['avg'], label='Avg')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.legend(loc="upper left")
        plt.show()

    env.close()

q_table, bins = Qtable(len(env.observation_space.low), env.action_space.n)

q_learning(q_table, bins, lr=0.15, gamma=0.5, episodes=10000, timestep=1000)
