#This is a double Deep Q network 
import random
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

class DDQN:
	def __init__(self,nS,nA):
		self.nS=nS #Number of observation/States
		self.nA=nA	#Number of action
		self.exploration_rate=1	 
		self.exploration_min=0.01
		self.exploration_decay=0.985
		self.gamma=0.99
		self.learning_rate=1e-4
		self.verbose=0
		self.minibatch_size=64
		self.memory=deque(maxlen=1000) #Expreience replay size
		self.bad_memory=deque(maxlen=1000)
		self.slow_memory=deque(maxlen=1000)
		self.model=self.create_model() #calls the create model
		self.target_model=self.create_model() #Target Network

	def create_model(self): #we do this to keep 2 networks
		model=Sequential() #the model here is a local variable
		model.add(Dense(16, input_shape=(self.nS,),activation="relu"))
		model.add(Dense(24,activation="relu"))
		# model.add(Dense(64,activation="relu"))
		model.add(Dense(self.nA,activation="linear"))
		model.compile(loss="mse",optimizer=Adam(lr=self.learning_rate))
		return model

	def target_model_update(self):
		self.target_model.set_weights(self.model.get_weights()) #set the weights of the target network using the weights of the original network 

	def act(self,state):
		if np.random.random()<self.exploration_rate:
			return np.random.choice(self.nA)
		q=self.model.predict(state)
		return np.argmax(q[0])

	def good_replay(self):
		#does the iteration using for loops		
		
		if(len(self.memory)>0):
			if(len(self.memory)<self.minibatch_size):
				minibatch=random.sample(self.memory,len(self.memory))
			else:
				minibatch=random.sample(self.memory,self.minibatch_size)
			for s,a,r,ns,done in minibatch:		
				q_update=r
				if not done:
					q_update=r+self.gamma*np.amax(self.target_model.predict(ns)[0])
				q_values=self.model.predict(s)
				q_values[0][a]=q_update
				# print(self.model.predict(s)[0][a],q_update)
				self.model.fit(s, q_values, epochs=1, verbose=self.verbose)

	def bad_replay(self):
        
		if(len(self.bad_memory)>0):
			if(len(self.bad_memory)<self.minibatch_size):
				minibatch1=random.sample(self.bad_memory,len(self.bad_memory))
			else:
				minibatch1=random.sample(self.bad_memory,self.minibatch_size)
			for s,a,r,ns,done in minibatch1:			
				q_update=r
				if not done:
					q_update=r+self.gamma*np.amax(self.target_model.predict(ns)[0])
				q_values=self.model.predict(s)
				q_values[0][a]=q_update
				# print(self.model.predict(s)[0][a],q_update)
				self.model.fit(s, q_values, epochs=1, verbose=self.verbose)

	def slow_replay(self):
		#does the iteration using for loops		
		
		if(len(self.slow_memory)>0):
			if(len(self.slow_memory)<self.minibatch_size):
				minibatch=random.sample(self.slow_memory,len(self.slow_memory))
			else:
				minibatch=random.sample(self.slow_memory,self.minibatch_size)
			for s,a,r,ns,done in minibatch:			
				q_update=r
				if not done:
					q_update=r+self.gamma*np.amax(self.target_model.predict(ns)[0])
				q_values=self.model.predict(s)
				q_values[0][a]=q_update
				# print(self.model.predict(s)[0][a],q_update)
				self.model.fit(s, q_values, epochs=1, verbose=self.verbose)