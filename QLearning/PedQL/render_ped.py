# Implementing Q learning On The Inverted Pendulum Problem. 
# Reference: https://github.tamu.edu/desik-rengarajan/IRL

import pickle
with open('ped2_latest_small.pickle', 'rb') as f:
    q_table = pickle.load(f)
    # print(sum(q_table))

from ped_car import PedestrianEnv
import numpy as np
import random
import math
from time import sleep
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import pdb
# %matplotlib inline

## Initialize the "Pedestrian" environment
env = PedestrianEnv()
# observation_space_low = [0,0,1.5,10,-3]
# observation_space_high = [10,75,7.5,15,3]
observation_space_low = [0,0,2.5,10,-3]
observation_space_high = [10,60,10,15,3]

## Defining the environment related constants

# Number of discrete states and actions (bucket) per dimension
NUM_BUCKETS = (21,61,4,21,13)  # (p_y, c_x, c_y, c_v, c_a) = (40, 120, 3, 40, 24) add 1 to all!
NUM_ACTIONS = 5

# bounds for each discrete state
STATE_BOUNDS = list(zip(observation_space_low, observation_space_high))

# bounds for action and state spaces
action_space_low = -2
action_space_high = 2
action_bins = np.squeeze(np.linspace(action_space_low, action_space_high, NUM_ACTIONS))
state_bins = []
for i in range(5):
    state_bins.append(np.linspace(STATE_BOUNDS[i][0], STATE_BOUNDS[i][1], NUM_BUCKETS[i]-1))

## Defining the simulation related constants
NUM_EPISODES = 25000
# MAX_T = 200
DEBUG_MODE = False

# ## Creating a Q-Table for each state-action pair
# q_table = np.zeros(NUM_BUCKETS + (NUM_ACTIONS,))

def select_action(state, explore_rate):
    # Select a random action
    # if random.random() < explore_rate:
    #     action = env.action_space.sample()
    # # Select the action with the highest q
    # else:
    action = np.argmax(q_table[state])
    action = map_action(action)
    return action

def map_action(action_idx):
    if(action_idx == 0):
        lower_limit = action_space_low
        upper_limit = action_space_low
    elif(action_idx == NUM_ACTIONS):
        lower_limit = action_space_high
        upper_limit = action_space_high
    else:
        lower_limit = action_bins[action_idx-1]
        upper_limit = action_bins[action_idx]
    
    action = np.random.uniform(low=lower_limit, high=upper_limit, size=1)
    return action

def bucket_action(action):
    return np.digitize(action, action_bins, right=True)

def bucket_state(states):
    idx = []
    for i, state in enumerate(states):
        idx.append(np.digitize(state, state_bins[i], right=True))
    return tuple(idx)


## Instantiating the learning related parameters
learning_rate = 1
explore_rate = 1
decay_rate_exp = 0.0002
decay_rate_lea = 0.0002

max_explore_rate = 1
max_learn_rate = 1
min_explore_rate = 0.01
min_learn_rate = 0.1

discount_factor = 0.99
rew = np.zeros(NUM_EPISODES)
Rate_explore = np.zeros((NUM_EPISODES,1))
Rate_learn = np.zeros((NUM_EPISODES,1))

Ped_Pos=[]
Car_xPos=[]
Car_yPos=[]
d = env.d
W = env.W

death_toll=0
safe_chicken=0
done_count=0
count=0

episodes = 10
for episode in range(episodes):
    c_state = bucket_state(env.reset(np.random.randint(1,4)))
    cum_rew_ep = 0

    while True:
        action = select_action(c_state, explore_rate)
        # Execute the action
        n_state, reward, done = env.step(action)
        # n_state = np.array(n_state)

        # print(n_state)
        # print(reward)
        Ped_Pos.append(n_state[0])
        Car_xPos.append(n_state[1])
        Car_yPos.append(n_state[2])

        # bucket states and action
        action = bucket_action(action)
        print(action)
        n_state = bucket_state(n_state)

        best_q = np.amax(q_table[n_state])

        # Setting up for the next iteration
        c_state = n_state                  
        cum_rew_ep += reward

        count = count+1
        if done:
            done_count+=1
            if (reward==-10000):
                death_toll+=1               
            if (reward==5000):
                safe_chicken+=1
            break

    Rate_explore[episode] = explore_rate
    Rate_learn[episode] = learning_rate

    # Update parameters
    rew[episode] = cum_rew_ep

    #Exponential learning
    decay_parameter_exp = np.exp(-decay_rate_exp * (episode+1))
    decay_parameter_lea = np.exp(-decay_rate_lea * (episode+1))
    explore_rate = min_explore_rate + (max_explore_rate - min_explore_rate)*decay_parameter_exp
    learning_rate = min_learn_rate + (max_learn_rate - min_learn_rate)*decay_parameter_lea

print(Ped_Pos)
# print(Car_xPos)
# print(Car_yPos)

#Plot initialization
fig, ax = plt.subplots()
ax.set_xlim(0, 1.8*d)
ax.set_ylim(0, W+2)
xdata, ydata = [], []
ln1, = plt.plot([], [], 'ro',markersize=20)
ln2, = plt.plot([], [], 'bx',markersize=20)

#Animate Car
def update1(frame1):
    ln1.set_data(Car_xPos[frame1], Car_yPos[frame1])
    return ln1,

#Animate Ped
def update2(frame2):
    ln2.set_data(d, Ped_Pos[frame2])
    return ln2,

def updateALL(frame):
    a = update1(frame)
    b = update2(frame)
    print(frame)
    return a+b

#Animate
iterations=len(Ped_Pos)
plt.grid()
ani = FuncAnimation(fig, updateALL, frames= iterations,
                    blit=True,repeat=False)
plt.pause(0.25)
plt.show()

#Results
print('Episodes', done_count)
print('Safe_chicken',safe_chicken)
print('Death_toll '+str(death_toll))
print('Did_not_reach '+str(done_count-safe_chicken-death_toll))
print('Death_toll % '+str(death_toll*100/done_count))

ani.save('animation_render_ped_latest.gif', writer='imagemagick', fps=120)