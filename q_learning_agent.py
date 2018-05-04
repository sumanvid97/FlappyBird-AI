import os
import random
import numpy as np

class QLearningAgent(object):

	def __init__(self,train):
		self.train = train
		self.episode = 0
		self.discount_factor = 0.95
		self.learning_rate = 0.7
		self.previous_state = [96,47,0]
		self.previous_action = 0
		self.moves = []
		self.scores = []
		self.max_score = 0
		self.xdim = 130
		self.ydim = 130
		self.vdim = 20
		self.qvalues = np.zeros((self.xdim, self.ydim, self.vdim, 2))
		self.initialize_model()

	def initialize_model(self):
		if os.path.exists("qvalues.txt"):
			qfile = open("qvalues.txt","r")
			line = qfile.readline()
			if self.train:
				self.episode = int(line)
			line = qfile.readline()
			while len(line) != 0:
				state = line.split(',')
				self.qvalues[int(state[0]),int(state[1]),int(state[2]),int(state[3])] = float(state[4])
				line = qfile.readline()
			qfile.close()
		
	def act(self, xdist, ydist, vely):
		if self.train:
			state = [xdist,ydist,vely]
			self.moves.append([self.previous_state,self.previous_action,state,0])
			self.previous_state = state

		if self.qvalues[xdist,ydist,vely][0] >= self.qvalues[xdist,ydist,vely][1]:
			self.previous_action = 0
		else:
			self.previous_action = 1
		return self.previous_action

	def record(self,reward):
		self.moves[-1][3] = reward

	def update_qvalues(self, score):
		self.episode += 1
		self.max_score = max(self.max_score, score)
		print("Episode: " + str(self.episode) + " Score: " + str(score) + " Max Score: " + str(self.max_score))
		self.scores.append(score)
		
		if self.train:
			history = list(reversed(self.moves))			
			first = True
			second = True
			jump = True
			if history[0][1] < 69:
				jump = False
			for move in history:
				[x,y,v] = move[0]
				action = move[1]
				[x1,y1,z1] = move[2]
				reward = move[3]
				if first or second:
					reward = -1000000
					if first:
						first = False
					else:
						second = False
				if jump and action:
					reward = -1000000
					jump = False
				self.qvalues[x,y,v,action] = (1- self.learning_rate) * (self.qvalues[x,y,v,action]) + (self.learning_rate) * ( reward + (self.discount_factor)*max(self.qvalues[x1,y1,z1,0],self.qvalues[x1,y1,z1,1]))
			self.moves = []

	def save_model(self):
		data = str(self.episode) + "\n"
		for x in range(self.xdim):
			for y in range(self.ydim):
				for v in range(self.vdim):
					for a in range(2):
						data += str(x) + ", " + str(y) + ", " + str(v) + ", " + str(a) + ", " + str(self.qvalues[x,y,v,a]) + "\n"
		qfile = open("qvalues.txt","w")
		qfile.write(data)
		qfile.close()
		
		data1 = ''
		for i in range(len(self.scores)):
			data1 += str(self.scores[i]) + "\n"
		sfile = open("scores.txt","a+")
		sfile.write(data1)
		sfile.close() 

class QLearningAgentGreedy(object):

	def __init__(self,train):
		self.train = train
		self.episode = 0
		self.discount_factor = 0.95
		self.learning_rate = 0.7
		self.previous_state = [96,47,0]
		self.previous_action = 0
		self.epsilon = 0.1
		self.final_epsilon = 0.0
		self.epsilon_decay = 0.00001
		self.max_score = 0
		self.xdim = 130
		self.ydim = 130
		self.vdim = 20
		self.moves = []
		self.scores = []
		self.qvalues = np.zeros((self.xdim, self.ydim, self.vdim, 2))
		self.initialize_model()

	def initialize_model(self):
		if os.path.exists("qvalues_greedy.txt"):
			qfile = open("qvalues_greedy.txt","r")
			line = qfile.readline()
			if self.train:
				[self.episode,self.epsilon] = [int(line.split(',')[0]), float(line.split(',')[1])]
			line = qfile.readline()
			while len(line) != 0:
				state = line.split(',')
				self.qvalues[int(state[0]),int(state[1]),int(state[2]),int(state[3])] = float(state[4])
				line = qfile.readline()
			qfile.close()
		
	def act(self, xdist, ydist, vely):
		if self.train:
			state = [xdist,ydist,vely]
			self.moves.append([self.previous_state,self.previous_action,state,0])
			self.previous_state = state

			if random.random() <= self.epsilon:
				self.previous_action = random.randrange(2)
			elif self.qvalues[xdist,ydist,vely][0] >= self.qvalues[xdist,ydist,vely][1]:
				self.previous_action = 0
			else:
				self.previous_action = 1
		else:
			if self.qvalues[xdist,ydist,vely][0] >= self.qvalues[xdist,ydist,vely][1]:
				self.previous_action = 0
			else:
				self.previous_action = 1
		
		return self.previous_action

	def record(self,reward):
		self.moves[-1][3] = reward

	def update_qvalues(self, score):
		self.episode += 1
		self.max_score = max(self.max_score, score)
		print("Episode: " + str(self.episode) + " Epsilon: " + str(self.epsilon) + " Score: " + str(score) + " Max Score: " + str(self.max_score))
		self.scores.append(score)
		
		if self.train:
			history = list(reversed(self.moves))
			first = True
			second = True
			jump = True
			if history[0][1] < 69:
				jump = False
			for move in history:
				[x,y,v] = move[0]
				action = move[1]
				[x1,y1,z1] = move[2]
				reward = move[3]
				if first or second:
					reward = -1
					if first:
						first = False
					else:
						second = False
				if jump and action:
					reward = -1
					jump = False
				self.qvalues[x,y,v,action] = (1- self.learning_rate) * (self.qvalues[x,y,v,action]) + (self.learning_rate) * ( reward + (self.discount_factor)*max(self.qvalues[x1,y1,z1,0],self.qvalues[x1,y1,z1,1]))

			self.moves = []
			if self.epsilon > self.final_epsilon:
				self.epsilon -= self.epsilon_decay
		
	def save_model(self):
		data = str(self.episode) + "," + str(self.epsilon) + "\n"
		for x in range(self.xdim):
			for y in range(self.ydim):
				for v in range(self.vdim):
					for a in range(2):
						data += str(x) + ", " + str(y) + ", " + str(v) + ", " + str(a) + ", " + str(self.qvalues[x,y,v,a]) + "\n"
		qfile = open("qvalues_greedy.txt","w")
		qfile.write(data)
		qfile.close()
		
		data1 = ''
		for i in range(len(self.scores)):
			data1 += str(self.scores[i]) + "\n"
		sfile = open("scores_greedy.txt","a+")
		sfile.write(data1)
		sfile.close() 
