"""
Pedestrian Environment
Texas A&M University
LENS Group 2019
"""
import math
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np
import random
import pickle


class CarEnv(gym.Env):

	"""
	Description:
		A pedestrian aims to cross the road without being hit by the autonomous car. Goal is to make the pedestrian reach the other end safely.
	Observation:
		Parameters
		Width of the road W=9
		Length of observation D=40
		Safe Region square box of scar_ped_2ize 1.5 around the pedestrian

		Type: Box(4)
		Num	Observation                          Min         Max
		0	Car Position (C_position)			[0,1.5]		[40,7.5]
		1	Car Velocity (C_velocity)			  0		  	  15
		2	Car acceleration (C_acceleration)	 -1.5		 1.5
		3	Pedestrian postition (P_position)    [40,0]      [40,9]

	Actions:
		Type: Discrete(2)
		Num	Action
		0	Move front with velocity v
		1	Move back with velocity -v

	#######Have to edit this portion.

		Reward is 1 for every step car_ped_2taken, including the termination step
	Starting State: ????
		All observations are assigned a uniform random value in [-0.05..0.05]

	Episode Termination: Check with Desik ?????
		The pedestrian is dead. RIP
		Pedestrian reaches end of the cross road of dimension W or Car crosses length d.

		Solved Requirements
		Considered solved when the pedestrian reaches the other side of the road safely.
	"""
	def __init__(self,W=10,d=50,C_velocity_limit=15,C_acceleration_limit=3,P_velocity_limit=2,safe_x=2,safe_y=1,safer_x=4,safer_y=3,prob=0.8,delta_t=0.25):
		self.d = d
		self.W = W
		self.state = None
		self.C_state = None
		self.P_state = None
		self.P_velocity_limit = P_velocity_limit
		self.C_velocity_limit = C_velocity_limit
		self.C_acceleration_limit = C_acceleration_limit
		self.action_space = spaces.Box(np.array([-1*self.C_acceleration_limit]),np.array([self.C_acceleration_limit]), dtype= np.float32)
		self.safe_x=safe_x
		self.safe_y=safe_y
		self.safer_x=safer_x
		self.safer_y=safer_y
		self.prob=prob
		self.delta_t=delta_t
		self.time=None


	def reset(self,flag):
		self.time=0
		self.P_state=[random.uniform(2.5,10)]
		if (flag==1):
			#lane 1
			self.C_state=[0,
						random.uniform(2.5,5),
						self.C_velocity_limit,
						random.uniform(0,self.C_acceleration_limit)]
		if (flag==2):
			#lane 2
			self.C_state=[0,
						random.uniform(5,7),self.C_velocity_limit,
						random.uniform(0,self.C_acceleration_limit)]
		if (flag==3):
			#lane 3
			self.C_state=[0,
						random.uniform(7,self.W - 1.0),self.C_velocity_limit,
						random.uniform(0,self.C_acceleration_limit)]

		self.state=self.P_state+self.C_state

		return self.state
	

	def step_C(self,state,action):

		# next state calculation
		C_acceleration_temp = action

		C_position_y_temp = state[2]
		
		C_position_x_temp = max(state[1],state[1]+state[3]*self.delta_t+0.5*C_acceleration_temp*self.delta_t**2)
		
		C_velocity_temp =min(max(0,state[3]+C_acceleration_temp*self.delta_t),self.C_velocity_limit)
		
		C_state_temp=[C_position_x_temp,
					C_position_y_temp,
					C_velocity_temp,
					C_acceleration_temp]
		
		
		done=False

		goal = 15.0 + self.d

		if (state[2] - self.safe_y) < state[0] < (state[2] + self.safe_y) and \
			 (self.d - self.safe_x) < state[1] < (self.d   + self.safe_x) :
			 # got hit
			reward = -500.0
			done=True

		elif(state[1] > goal):
			# reached goal
			reward = 75.0
			done=True

		else:
			
			# travel incentive
			R_travel = -0.00236*5*((goal + 2.0- state[1])**(2)) 

			if (self.d-20.0) < state[1] < (self.d-10.0) and state[0] < (state[2]+self.safe_y ) :
				
				if not (state[1] == C_position_x_temp):
					R_travel += -5.0

			# fear factor
			if state[1] < (self.d + self.safe_x) and state[0] < (state[2]+self.safe_y):

				R_fear = -20.0*( (state[3]/(self.C_velocity_limit+0.1))**(( (state[0]-state[2])**2 + (self.d-state[1])**2 ) **(0.5) ))

			else:
				R_fear = 0.0

			reward = R_fear + R_travel

		return C_state_temp,reward,done



	def step_P(self,state,action_ped):

		#discretization
		P_state_temp=min(self.W,max(0.0, state[0]+ action_ped*self.delta_t))
		
		return [P_state_temp]

	def step(self, action_car, action_ped):
		assert self.action_space.contains(action_car), "%r (%s) invalid"%(action_car, type(action_car))
		action_car = action_car[0]
		action_ped = action_ped[0]
		# timer update
		self.time+=1
		# print(self.state)
		# state definitiopn

		# step_car
		C_state_temp,reward,done = self.step_C(self.state,action_car)
		# step_ped
		P_state_temp = self.step_P(self.state,action_ped)
		
		self.C_state = C_state_temp
		self.P_state = P_state_temp
		self.state   = self.P_state+self.C_state

		# if crossed time terminate
		if(self.time>250):
			done=True
			reward=-200

		return self.state,reward,done
