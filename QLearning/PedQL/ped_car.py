"""
Pedestrian Environment v1.1

Texas A&M University
LENS Group 2019
In this env, the car does not follow random velocities
"""
import math
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np
import random

class PedestrianEnv(gym.Env):


	def __init__(self,W=10,d=50,C_velocity_limit=15,C_acceleration_limit=3,P_velocity_limit=2,safe_x=2,safe_y=1,safer_x=4,safer_y=3,prob=0.8,static_state_prob=0.1,delta_t=0.25):
		print("Reinitialized The car zones")
		self.d = d
		self.W = W
		self.state = None
		self.C_state = None
		self.P_state = None
		self.P_velocity_limit = P_velocity_limit 
		self.C_velocity_limit = C_velocity_limit
		self.C_acceleration_limit = C_acceleration_limit
		self.action_space = spaces.Box(np.array([-1*self.P_velocity_limit]),np.array([self.P_velocity_limit]), dtype= np.float32)
		self.safe_x=safe_x
		self.safe_y=safe_y
		self.safer_x=safer_x
		self.safer_y=safer_y
		self.stop_at_zebra_prob = prob
		self.static_state_prob = static_state_prob
		self.delta_t=delta_t
		self.time=None

	def reset(self,flag):
		self.time=0
		self.static_state = random.uniform(0,1) < self.static_state_prob
		self.P_state=[0]
		
		if not self.static_state: 
			if (flag==1):
				#lane 1
				self.C_state=[random.uniform(0,self.d+10),
							random.uniform(2.5,5),
							random.uniform(10,self.C_velocity_limit),
							random.uniform(0,self.C_acceleration_limit)]
			if (flag==2):
				#lane 2
				self.C_state=[random.uniform(0,self.d+10),
							random.uniform(5,7),
							random.uniform(10,self.C_velocity_limit),
							random.uniform(0,self.C_acceleration_limit)]
			if (flag==3):
				#lane 3
				self.C_state=[random.uniform(0,self.d+10),
							random.uniform(7,self.W - 1.0),
							random.uniform(10,self.C_velocity_limit),
							random.uniform(0,self.C_acceleration_limit)]
		else:
			self.C_state=[random.uniform(0,self.d-self.safe_x),
						random.uniform(2.5,self.W - 1.0),0,0]

		self.state = self.P_state+self.C_state
		
		return self.state,self.static_state

	def step_C(self,state,action):
		
		if (not self.static_state):
			

			if (((state[2] - self.safer_y ) < state[0] < (state[2] + self.safer_y ) and \
						 	(self.d - self.safer_x ) < state[1] < (self.d- self.safe_x))
                    or \
							((self.d-20.0) < state[1] < (self.d-10.0) and (state[0] < (state[2]+self.safe_y)) and \
								(random.uniform(0,1) < self.stop_at_zebra_prob))
							):

				C_acceleration_temp=0
				C_velocity_temp=0

			
			elif state[0] < (state[2] + self.safe_y) and \
			 	(self.d - 10.0) < state[1] < (self.d) :

				C_acceleration_temp = random.uniform(0,self.C_acceleration_limit)
				C_velocity_temp 	= min(max(0,state[3]+C_acceleration_temp*self.delta_t),self.C_velocity_limit/2.0)

			else:
				C_acceleration_temp = random.uniform(-1.0*self.C_acceleration_limit,self.C_acceleration_limit)
				C_velocity_temp 	= min(max(0,state[3]+C_acceleration_temp*self.delta_t),self.C_velocity_limit)

		else:
			C_acceleration_temp = 0
			C_velocity_temp 	= 0
		
		C_position_y_temp = state[2]
		C_position_x_temp = max(state[1],state[1]+state[3]*self.delta_t+0.5*C_acceleration_temp*self.delta_t**2)
		C_state_temp=[C_position_x_temp,
				C_position_y_temp,
				C_velocity_temp,
				C_acceleration_temp]

		return C_state_temp

	def step_P(self,state,action):
		P_state_temp=min(self.W,max(0,state[0]+action*self.delta_t))
		done=False

		if (state[2] - self.safe_y) < state[0] < (state[2] + self.safe_y) and \
			 (self.d - self.safe_x) < state[1] < (self.d   + self.safe_x) :
			reward = (-100.0)
			done=True
			return [P_state_temp],reward,done

		elif(state[0]>=self.W):
			reward = 75.0
			done=True
			return [P_state_temp],reward,done

		else:
			R_travel = -0.1*((self.W + 2.0- state[0])**(2)) 

			if (self.d - 25.0) < state[1] < (self.d + self.safe_x):

				R_fear = -20.0*( (state[3]/(self.C_velocity_limit+0.1))  **
                                 (( (state[0]-state[2])**2 + (self.d-state[1])**2 )
                                  **(0.5) ))

			else:
				R_fear = 0.0

			# print(state)
			# print('R_fear: '+str(R_fear)+' R_travel: '+str(R_travel))

			reward= (R_travel) + (R_fear) 

			return [P_state_temp],reward,done

	def step(self, action_1):
		assert self.action_space.contains(action_1), "%r (%s) invalid"%(action_1, type(action_1))
		action=action_1[0]
		self.time+=1

		C_state_temp=self.step_C(self.state,action)
		P_state_temp,reward,done=self.step_P(self.state,action)
		self.C_state=C_state_temp
		self.P_state=P_state_temp
		self.state=self.P_state+self.C_state

		if(self.time>250):
			done=True
			reward=-50.0

		return self.state,reward,done