# Implementing Q learning . 

# Reference: https://github.tamu.edu/desik-rengarajan/IRL
from car_ped import CarEnv
import numpy as np
import random
import math
from time import sleep
import matplotlib.pyplot as plt
import pdb
import pickle
from matplotlib.animation import FuncAnimation


## Initialize the "Pedestrian" environment
env = CarEnv()

# (common)
observation_space_low = [0,0,2.5,0,-3]
observation_space_high = [10,60,10,15,3]

## Defining the environment related constants

# Number of discrete states and actions (bucket) per dimension
PED_NUM_BUCKETS = (21,61,4,21,13)  
CAR_NUM_BUCKETS = (21,61,4,21,13)  
PED_NUM_ACTIONS = 10
CAR_NUM_ACTIONS = 10

# bounds for each discrete state
# (common)
STATE_BOUNDS = list(zip(observation_space_low, observation_space_high))

# bounds for action and state spaces

#car
Car_action_space_low = -1*env.C_acceleration_limit
Car_action_space_high = 1*env.C_acceleration_limit
Car_action_bins = np.squeeze(np.linspace(Car_action_space_low, Car_action_space_high, CAR_NUM_ACTIONS))

#ped
Ped_action_space_low = -1*env.P_velocity_limit
Ped_action_space_high = 1*env.P_velocity_limit
Ped_action_bins = np.squeeze(np.linspace(Ped_action_space_low, Ped_action_space_high, PED_NUM_ACTIONS))

# (common)
state_bins = []
for i in range(5):
    state_bins.append(np.linspace(STATE_BOUNDS[i][0], STATE_BOUNDS[i][1], CAR_NUM_BUCKETS[i]-1))

## Defining the simulation related constants
NUM_EPISODES = 1500
DEBUG_MODE = False

## Q-Table for each state-action pair

#CAR
with open('CarQL.pickle', 'rb') as f:
    q_table_car = pickle.load(f)

#PED
with open('pedQL.pickle', 'rb') as f:
    q_table_ped = pickle.load(f)


##############################################################################################################
def map_action(action_idx, action_bins, action_space_low, action_space_high, NUM_ACTIONS):
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


def select_action(state, explore_rate ,q_table, action_bins, action_space_low, action_space_high,NUM_ACTIONS):
    # Select a random action
    if random.random() < explore_rate:
        action = env.action_space.sample()
    else:
        action = np.argmax(q_table[state])
        action = map_action(action, action_bins, action_space_low, action_space_high,NUM_ACTIONS)
        
    return action

def bucket_action(action,action_bins):
    return np.digitize(action, action_bins, right=True) #right is different

def bucket_state(states):
    idx = []
    for i, state in enumerate(states):
        idx.append(np.digitize(state, state_bins[i], right=True)) #right is different
    return tuple(idx)

##############################################################################################################
## Instantiating the learning related parameters

# Learning rate
learning_rate = 1
decay_rate_lea = 0.0002
max_learn_rate = 1
min_learn_rate = 0.01

# Explore rate
explore_rate = 1
decay_rate_exp = 0.00002
max_explore_rate = 1
min_explore_rate = 0.01

# discount factors
discount_factor = 0.99
rew = np.zeros(NUM_EPISODES)
Rate_explore = np.zeros((NUM_EPISODES,1))
Rate_learn = np.zeros((NUM_EPISODES,1))

death_toll=0
safe_chicken=0
done_count=0
count=0
actions_list = []



Ped_Pos=[]
Car_xPos=[]
Car_yPos=[]
d = env.d
W = env.W

for episode in range(NUM_EPISODES):
    
    c_state = bucket_state(env.reset(np.random.randint(1,2)))
    cum_rew_ep = 0
    
    while True:
        
        # Select an action
        
        action_car = select_action(c_state, 0.0 ,q_table_car, 
                                   Car_action_bins, Car_action_space_low, Car_action_space_high,CAR_NUM_ACTIONS)
        
        action_ped = select_action(c_state, 0.0, q_table_ped, 
                                   Ped_action_bins, Ped_action_space_low, Ped_action_space_high,PED_NUM_ACTIONS)
        
        # Execute the action
        
        n_state, reward, done = env.step(action_car,action_ped)
        
        temp = n_state
        

        Ped_Pos.append(n_state[0])
        Car_xPos.append(n_state[1])
        Car_yPos.append(n_state[2])
            
        # Setting up for the next iteration
        n_state = bucket_state(n_state)
        
        c_state = n_state
        
        cum_rew_ep += reward
        

        # Print data
        if done:
            
            print(temp)
            # print the episode number and remaning cumulative reward\
            print('EPISODE:'+str(episode)+'   REWARD: '+str(reward) + ' Cumulative Reward: '+str(cum_rew_ep)) 
            
            done_count+=1
            
            if (reward==-500):
                death_toll+=1               
            
            if (reward==75):
                safe_chicken+=1
            
            break
        



# print(Ped_Pos)
# print(Car_xPos)
# print(Car_yPos)

#Plot initialization
fig, ax = plt.subplots()
ax.set_xlim(0, 1.8*d)
ax.set_ylim(0, W+2)
xdata, ydata = [], []
ln1, = plt.plot([], [], 'ro',markersize=10)
ln2, = plt.plot([], [], 'bx',markersize=10)

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

#Results
print('Episodes', done_count)
print('Safe_chicken',safe_chicken)
print('Death_toll '+str(death_toll))
print('Did_not_reach '+str(done_count-safe_chicken-death_toll))
print('Death_toll % '+str(death_toll*100/done_count))
#Animate
iterations=len(Ped_Pos)
plt.grid()
ani = FuncAnimation(fig, updateALL, frames= iterations,
                    blit=True,repeat=False)
plt.pause(0.25)
plt.show()

