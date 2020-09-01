from ped_car_2 import PedestrianEnv
import numpy as np
from MySolver import DQNSolver
import random
import matplotlib.pyplot as plt
max_run=500
env=PedestrianEnv()
observation = env.reset()
observation_space=len(observation) #we get the number of parameters in the state
action_space=5 #number of discrete velocities pedestrian can take
dqn_solver = DQNSolver(observation_space, action_space)
run=0
plotter=[0]
while run<max_run:
	run += 1
	state = env.reset()     
	state = np.reshape(state, [1, observation_space])       
	step = 0
	rwd=0
	while True:     
		action = dqn_solver.act(state)              
		#action_1=-1*env.P_velocity_limit+action*(2*(env.P_velocity_limit)/(action_space-1))
		action_1=action*((env.P_velocity_limit)/(action_space-1))                       
		state_next, reward, terminal = env.step(np.array([action_1]))       
		rwd+=reward                     
		state_next = np.reshape(state_next, [1, observation_space])
		dqn_solver.remember(state, action, reward, state_next, terminal)
		state = state_next
		if terminal:
			print ("Run: " + str(run) + ", exploration: " + str(dqn_solver.exploration_rate) + ", score: " + str(rwd))
			plotter.append(plotter[run-1]+(rwd-plotter[run-1])/run)             
			break
		dqn_solver.experience_replay()


plt.figure(1)
plt.plot(plotter)
plt.savefig('reward_plot.png')



