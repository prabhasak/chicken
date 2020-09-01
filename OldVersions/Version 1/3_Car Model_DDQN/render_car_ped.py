#rendering the model
from car_ped_2 import CarEnv
import numpy as np
import random
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from keras.models import load_model
from My_DDQN import DDQN
env=CarEnv()
observation = env.reset()
observation_space=len(observation) #we get the number of parameters in the state
action_space=10 #number of discrete velocities pedestrian can take
agent=DDQN(observation_space, action_space)
agent.exploration_rate=0
agent.model=load_model('ddqn_car_Learner_1.h5')
death_toll=0
safe_chicken=0
safe_car=0
done_count=0
count=0
Ped_Pos=[]
Car_xPos=[]
Car_yPos=[]
d = env.d
W = env.W

env = CarEnv()
episodes = 3
for e in range(episodes):
	state=env.reset()
	state = np.reshape(state, [1, observation_space])
	while True:	
		action=agent.act(state)				
		action_1=-1*env.C_acceleration_limit+action*(2*(env.C_acceleration_limit)/(action_space-1))
		#action_1=0
		state_next, reward, done = env.step(np.array([action_1]))		
		Ped_Pos.append(state_next[0])
		Car_xPos.append(state_next[1])
		Car_yPos.append(state_next[2])
		state_next = np.reshape(state_next, [1, observation_space])
		state=state_next	    
		if done:
			done_count+=1
			if (reward==-10000):
				death_toll+=1				
			if (reward==5000):
				safe_car+=1
				if (env.state[0]>W):
					safe_chicken+=1
			break					
			

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
plt.pause(1)
plt.show()

#Results

ani.save('animation_dual_ped_pause.gif', writer='imagemagick', fps=60)