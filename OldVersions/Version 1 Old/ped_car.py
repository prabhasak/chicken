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
	def __init__(self,W=9,d=40,C_velocity_limit=15,C_acceleration_limit=1.5,P_velocity_limit=1.5,safe_x=0.5,safe_y=0.5,safer_x=1,safer_y=1,prob=0.8,delta_t=1):
		self.d = d
		self.W = W
		self.P_position = None 
		self.C_velocity = None
		self.C_position_x = None
		self.C_position_y = None	
		self.C_acceleration = None
		self.state = None
		self.C_state = None
		self.P_state = None		
		self.P_position_limit = W+1 
		self.C_position_x_limit = d+1 
		self.C_position_y_limit = W+1
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
		self.P_position=0
		self.C_position_x=0
		self.C_position_y=random.choice([1.5,4.5,7.5])
		self.C_velocity=random.uniform(0,self.C_velocity_limit)
		self.C_acceleration=random.uniform(-1*self.C_acceleration_limit,self.C_acceleration_limit)
		self.C_state=[self.C_position_x,
					self.C_position_y,
					self.C_velocity,
					self.C_acceleration]		
		self.P_state=[self.P_position]
		self.state=self.P_state+self.C_state
		return self.state	

	def step_C(self):		
		Time_CarToBox = (self.d-self.safe_x-self.C_position_x)/self.C_velocity
		Time_PedInBox_min = min(abs((self.C_position_y-(self.P_position+self.safe_y))/self.P_velocity),abs((self.C_position_y-self.P_position-self.safe_y)/self.P_velocity))
		Time_PedInBox_max = max(abs((self.C_position_y-(self.P_position+self.safe_y))/self.P_velocity),abs((self.C_position_y-self.P_position-self.safe_y)/self.P_velocity))
		if Time_CarToBox > Time_PedInBox_min and Time_CarToBox < Time_PedInBox_max : 
				if (random.uniform(0,1)<self.prob):
					C_acceleration_temp = random.uniform(-1*self.C_acceleration_limit,self.C_acceleration_limit)
				else:
					C_acceleration_temp=0					
		else:
			C_acceleration_temp = random.uniform(-1*self.C_acceleration_limit,self.C_acceleration_limit)
		C_position_y_temp = self.C_position_y
		C_position_x_temp = self.C_position_x+self.C_velocity*self.delta_t+0.5*C_acceleration_temp*self.delta_t**2
		C_velocity_temp = min(0,self.C_velocity+C_acceleration_temp*self.delta_t)
		C_state_temp=[C_position_x_temp,
					C_position_y_temp,
					C_velocity_temp,
					C_acceleration_temp]

		return C_state_temp

	def step_P(self,action):
		P_state_temp=self.P_position+action*self.delta_t
		done=False
		if(self.P_position>self.C_position_y-self.safe_y and
			self.P_position<self.C_position_y+self.safe_y and
			self.C_position_x>self.d-self.safe_x and
			self.C_position_x<self.d+self.safe_x) :
			reward=-1000
			done=True
			return [P_state_temp],reward,done
		elif(self.P_position>=self.W):
			reward=200
			done=True
			return [P_state_temp],reward,done
		else:
			R_intent=(self.P_position-self.W)/(abs(self.P_velocity)) 
			if(self.P_position>self.C_position_y-self.safer_y and
				self.P_position<self.C_position_y+self.safer_y and
				self.C_position_x>self.d-self.safer_x and
				self.C_position_x<self.d+self.safer_x):
				R_fear=-1*self.C_velocity-((self.P_position-self.C_position_y)**2+(self.d-self.C_position_x)**2)**0.5				
			else:
				R_fear=0
			reward= 2 * R_intent + R_fear
			return [P_state_temp],reward,done

	def step(self, action_1):		
		assert self.action_space.contains(action_1), "%r (%s) invalid"%(action_1, type(action_1))
		action=action_1[0]
		self.P_velocity=action
		self.time+=1
		C_state_temp=self.step_C()
		P_state_temp,reward,done=self.step_P(action)
		self.C_state=C_state_temp
		self.P_state=P_state_temp
		self.state=self.P_state+self.C_state
		return self.state,reward,done

			# if(state[3]==0):
		# 	Time_CarToBox=10000
		# else:		
		# 	Time_CarToBox = (self.d-self.safe_x-state[1])/state[3]
		# Time_PedInBox_min = min(abs((state[2]-(state[0]+self.safe_y))/action),abs((state[2]-state[0]-self.safe_y)/action))
		# Time_PedInBox_max = max(abs((state[2]-(state[0]+self.safe_y))/action),abs((state[2]-state[0]-self.safe_y)/action))
		# if Time_CarToBox > Time_PedInBox_min and Time_CarToBox < Time_PedInBox_max : 
		# 		if (random.uniform(0,1)<self.prob):
		# 			C_acceleration_temp = random.uniform(0,self.C_acceleration_limit)
		# 		else:
		# 			C_acceleration_temp=0					
		# else: