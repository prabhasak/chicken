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
		self.exploration_decay=0.995
		self.gamma=0.999
		self.learning_rate=1e-3
		self.verbose=0
		self.minibatch_size=32
		self.memory=deque(maxlen=500) #Expreience replay size
		self.bad_memory=deque(maxlen=500)
		self.model=self.create_model() #calls the create model
		self.target_model=self.create_model() #Target Network
	
	def create_model(self): #we do this to keep 2 networks
		model=Sequential() #the model here is a local variable
		model.add(Dense(256, input_shape=(self.nS,),activation="relu"))
		model.add(Dense(128,activation="relu"))
		model.add(Dense(64,activation="relu"))
		model.add(Dense(self.nA,activation="linear"))
		model.compile(loss="mse",optimizer=Adam(lr=self.learning_rate))
		return model

	def add_memory(self, state, action, reward, next_state, done):
		self.memory.append((state, action, reward, next_state, done)) #adding to the exprience replay memory

	def add_bad_memory(self, state, action, reward, next_state, done):
		self.bad_memory.append((state, action, reward, next_state, done))


	def target_model_update(self):
		self.target_model.set_weights(self.model.get_weights()) #set the weights of the target network using the weights of the original network 

	def act(self,state):
		if np.random.rand()<self.exploration_rate:
			return np.random.choice(self.nA)
		q=self.model.predict(state)
		return np.argmax(q[0])

	def replay(self):
		#does the iteration using for loops				
		minibatch=random.sample(self.memory,self.minibatch_size)
		for s,a,r,ns,done in minibatch:			
			q_update=r
			if not done:
				q_update=r+self.gamma*np.amax(self.target_model.predict(ns)[0])
			q_values=self.model.predict(s)
			q_values[0][a]=q_update
			self.model.fit(s, q_values, verbose=self.verbose)

		if(len(self.bad_memory)>0):
			if(len(self.bad_memory)<self.minibatch_size):
				minibatch1=random.sample(self.bad_memory,len(self.bad_memory))
			else:
				minibatch1=random.sample(self.bad_memory,len(self.bad_memory))
			for s,a,r,ns,done in minibatch1:			
				q_update=r
				if not done:
					q_update=r+self.gamma*np.amax(self.target_model.predict(ns)[0])
				q_values=self.model.predict(s)
				q_values[0][a]=q_update
				self.model.fit(s, q_values, verbose=self.verbose)

		#self.exploration_rate*=self.exploration_decay
		#self.exploration_rate=max(self.exploration_rate,self.exploration_min)