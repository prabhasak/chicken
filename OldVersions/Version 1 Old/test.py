from ped_car_2 import PedestrianEnv
import numpy as np
import random
env = PedestrianEnv()
observation = env.reset()
death_toll=0
safe_chicken=0
done_count=0
rwd=0   
for t in range(100000): 
	action = np.array([1.3])   
	observation, reward, done = env.step(action)
	rwd+=reward
	#print(observation, reward, done)   
	if done:
		# print(rwd)
		done_count+=1
		if (reward==-10000):
			death_toll+=1
			#print(env.time)
			#print(rwd)     
									
		if (reward==5000):
			# print(env.time)
			#print(rwd)
			safe_chicken+=1
						

		rwd=0
		env.reset()
print('Death_toll '+str(death_toll))
print('Episodes', done_count)
print('Death_toll % '+str(death_toll*100/(done_count)))
print('Safe_chicken',safe_chicken)
print('Safe_toll % '+str(safe_chicken*100/(done_count)))
	