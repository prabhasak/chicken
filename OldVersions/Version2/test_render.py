from ped_car_v11 import PedestrianEnv
import numpy as np
import random
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation

env = PedestrianEnv()
observation = env.reset()

#Intitialization
death_toll=0
safe_chicken=0
done_count=0
count=0
Ped_Pos=[]
Car_xPos=[]
Car_yPos=[]
d = env.d
W = env.W
iterations = 10000

#Main
for t in range(iterations):	
	action = np.array([random.uniform(0.5,1.5)])
	#print(action)
	observation, reward, done = env.step(action)
#	print(observation, reward, done)
    
#	if not done:
	Ped_Pos.append(observation[0])
	Car_xPos.append(observation[1])
	Car_yPos.append(observation[2])
    
	if done:
		done_count+=1
		if (reward==-1000):
			death_toll+=1
			#print(observation, reward, done)
		if (reward==200):
			safe_chicken+=1
			#print(observation, reward, done)
		env.reset()

#Plot initialization
fig, ax = plt.subplots()
ax.set_xlim(0, d+2)
ax.set_ylim(0, W)
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
plt.grid()
ani = FuncAnimation(fig, updateALL, frames= iterations,
                    blit=True,repeat=False)
plt.pause(1)
plt.show()

#Results
print('Episodes', done_count)
print('Safe_chicken',safe_chicken)
print('Death_toll '+str(death_toll))
print('Death_toll % '+str(death_toll*100/(death_toll+safe_chicken)))