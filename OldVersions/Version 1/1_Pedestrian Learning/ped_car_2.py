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

class PedestrianEnv(gym.Env):

	"""
	Description:
		A pedestrian aims to cross the road without being hit by the autonomous car. Goal is to make the pedestrian reach the other end safely.
	Observation: 
		Parameters
		Width of the road W=9
		Length of observation D=40
		Safe Region square box of size 1.5 around the pedestrian 

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

		Reward is 1 for every step taken, including the termination step
	Starting State: ????
		All observations are assigned a uniform random value in [-0.05..0.05]

	Episode Termination: Check with Desik ?????
		The pedestrian is dead. RIP
		Pedestrian reaches end of the cross road of dimension W or Car crosses length d.
		
		Solved Requirements
		Considered solved when the pedestrian reaches the other side of the road safely. 
	"""
	def __init__(self,W=10,d=50,C_velocity_limit=8,C_acceleration_limit=1.5,P_velocity_limit=1.5,safe_x=4,safe_y=1.5,safer_x=6,safer_y=2,prob=0.8,delta_t=0.1):
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
		self.prob=prob
		self.delta_t=delta_t
		self.time=None

	def reset(self):
		self.time=0		
		'''
		self.C_state=[self.C_position_x,
					state[2],
					self.C_velocity,
					self.C_acceleration]
		self.P_state=[state[0]]		'''		
		self.P_state=[0]
		# self.C_state=[0,
		# 			random.uniform(0+self.safer_y,self.W-self.safer_y),
		# 			random.uniform(10,self.C_velocity_limit),
		# 			random.uniform(-1*self.C_acceleration_limit,self.C_acceleration_limit)]	
		self.C_state=[random.uniform(0,1*self.d),
					random.uniform(0+self.safer_y,self.W-self.safer_y),
					random.uniform(10,self.C_velocity_limit),
					random.uniform(-1*self.C_acceleration_limit,self.C_acceleration_limit)]							
		self.state=self.P_state+self.C_state		
		return self.state	

	def step_C(self,state,action):	
		C_acceleration_temp = random.uniform(-1*self.C_acceleration_limit,self.C_acceleration_limit)
		C_position_y_temp = state[2]
		C_position_x_temp = state[1]+state[3]*self.delta_t+0.5*C_acceleration_temp*self.delta_t**2
		C_velocity_temp =min(max(0,state[3]+C_acceleration_temp*self.delta_t),self.C_velocity_limit)
		C_state_temp=[C_position_x_temp,
					C_position_y_temp,
					C_velocity_temp,
					C_acceleration_temp]
		return C_state_temp

	def step_P(self,state,action):
		P_state_temp=min(self.W,max(0,state[0]+action*self.delta_t))
		done=False
		if(state[0]>state[2]-self.safe_y and
			state[0]<state[2]+self.safe_y and
			state[1]>self.d-self.safe_x and
			state[1]<self.d+self.safe_x) :
			reward=-10000
			done=True			
			return [P_state_temp],reward,done
		elif(state[0]>=self.W):
			reward=5000
			done=True			
			return [P_state_temp],reward,done
		else:
			R_intent=-1*self.delta_t*self.time 
			if(state[0]>state[2]-self.safer_y and
				state[0]<state[2]+self.safer_y and
				state[1]>self.d-self.safer_x and
				state[1]<self.d+self.safer_x):
				R_fear=-1*state[3]-((state[0]-state[2])**2+(self.d-state[1])**2)**0.5				
			else:
				R_fear=0
			reward= R_intent + 0.5*R_fear			
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

		if(self.time>200):
			done=True
			reward=-1000
		
		return self.state,reward,done