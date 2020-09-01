#Generate Trajectories 
from ped_car_2 import PedestrianEnv
import numpy as np
import random
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from keras.models import load_model
from My_DDQN import DDQN
import pandas as pd
env=PedestrianEnv()
observation = env.reset()
observation_space=len(observation) #we get the number of parameters in the state
action_space=10 #number of discrete velocities pedestrian can take
agent=DDQN(observation_space, action_space)
agent.exploration_rate=0
agent.model=load_model('ddqn_ped_Learner.h5')
death_toll=0
safe_chicken=0
done_count=0
count=0
Ped_y=[]
Car_x=[]
Car_y=[]
Car_v=[]
Car_a=[]
action_list=[]
env = PedestrianEnv()
episodes = 300
for e in range(episodes):
	state=env.reset()
	state = np.reshape(state, [1, observation_space])
	while True:	
		action=agent.act(state)
		Ped_y.append(state[0][0])
		Car_x.append(state[0][1])
		Car_y.append(state[0][2])
		Car_v.append(state[0][3])
		Car_a.append(state[0][4])
		action_list.append(action)						
		action_1=-1*env.P_velocity_limit+action*(2*(env.P_velocity_limit)/(action_space-1))		
		state_next, reward, done = env.step(np.array([action_1]))		
		state_next = np.reshape(state_next, [1, observation_space])
		state=state_next					    
		if done:
			done_count+=1
			if (reward==-10000):
				death_toll+=1				
			if (reward==5000):
				safe_chicken+=1
			break

df=pd.DataFrame({'Ped_y':(Ped_y),'Car_x':(Car_x),'Car_y':(Car_y),'Car_v':(Car_v),'Car_a':(Car_a),'Action':(action_list)})
df.to_csv('Trajectories.csv', index=False)	
print('Episodes', done_count)
print('Safe_chicken',safe_chicken)
print('Death_toll '+str(death_toll))
print('Death_toll % '+str(death_toll*100/(death_toll+safe_chicken)))