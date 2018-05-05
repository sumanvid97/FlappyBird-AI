from flappy_dqn import State
import os
import sys
import random
import numpy as np
from collections import deque

import tensorflow
from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.core import Dense, Activation, Flatten
from keras.optimizers import Adam

import skimage as skimage
from skimage import transform, color, exposure
from skimage.transform import rotate

num_actions = 2 # number of valid actions
discount = 0.99 # decay rate of past observations
observe = 3200 # timesteps to observe before training
explore = 3000000 # frames over which to anneal epsilon
FINAL_EPSILON = 0.0001 # final value of epsilon
INITIAL_EPSILON = 0.1 # starting value of epsilon
replay_memory = 50000 # number of previous transitions to remember

def build_network():

	print ("Initializing model ....")
	model = Sequential()
	model.add(Conv2D(32, (8, 8), padding='same', strides=(4, 4), input_shape=(80,80,4)))
	model.add(Activation('relu'))
	model.add(Conv2D(64, (4, 4), padding='same', strides=(2, 2)))
	model.add(Activation('relu'))
	model.add(Conv2D(64, (3, 3), padding='same', strides=(1, 1)))
	model.add(Activation('relu'))
	model.add(Flatten())
	model.add(Dense(512))
	model.add(Activation('relu'))
	model.add(Dense(num_actions))

	if os.path.exists("dqn.h5"):
		print ("Loading weights from dqn.h5 .....")
		model.load_weights("dqn.h5")
		print ("Weights loaded successfully.")
	adam = Adam(lr=1e-4)
	model.compile(loss='mse',optimizer=adam)
	print ("Finished building model.")

	return model

def process(input):
	# convert the input from rgb to grey
	image = skimage.color.rgb2gray(input)
	# resize image to 80x80 from 288x404
	image = skimage.transform.resize(image,(80,80), mode='constant')
	# return image after stretching or shrinking its intensity levels
	image = skimage.exposure.rescale_intensity(image,out_range=(0,255))
	# scale down pixels values to (0,1)
	image = image / 255.0
	return image

def train_network(model,mode):
    if mode == 'Run':
    	train = False
    elif mode == 'Train':
    	train = True

    if train:
    	epsilon = INITIAL_EPSILON
    else:
    	epsilon = FINAL_EPSILON

    sfile = open("scores_dqn.txt","a+")
    episode = 1
    timestep = 0
    loss = 0
    # initialize an instance of game
    game = State()
    # store the previous observations in replay memory
    replay = deque()
    # take action 0 and get resultant state
    image, score, reward, alive = game.next(0)
    # preprocess the image and stack to 80x80x4 pixels
    image = process(image)
    input_image = np.stack((image, image, image, image), axis=2)
    input_image = input_image.reshape(1, input_image.shape[0], input_image.shape[1], input_image.shape[2])
    
    while (True):
        # get an action epsilon greedy policy
        if random.random() <= epsilon:
            action = random.randrange(num_actions)
        else:
            q = model.predict(input_image)       
            action = np.argmax(q)
        # decay epsilon linearly
        if epsilon > FINAL_EPSILON and timestep > observe:
            epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / explore
        # take selected action and get resultant state
        image1, score, reward, alive = game.next(action)
        # preprocess the image and stack to 80x80x4 pixels
        image1 = process(image1)
    	image1 = image1.reshape(1, image1.shape[0], image1.shape[1], 1) #1x80x80x1
        input_image1 = np.append(image1, input_image[:, :, :, :3], axis=3)

        if train:
        	# add current transition to replay buffer
	        replay.append((input_image, action, reward, input_image1, alive))
	        if len(replay) > replay_memory:
	            replay.popleft()

	        if timestep > observe:
	            # sample a minibatch of size 32 from replay memory
	            minibatch = random.sample(replay, 32)
	            s, a, r, s1, alive = zip(*minibatch)
	            s = np.concatenate(s)
	            s1 = np.concatenate(s1)
	            targets = model.predict(s)
	            targets[range(32), a] = r + discount*np.max(model.predict(s1), axis=1)*alive
	            loss += model.train_on_batch(s, targets)

        input_image = input_image1
        timestep = timestep + 1

        if train:
        	# save the weights after every 1000 timesteps
        	if timestep % 1000 == 0:
        		model.save_weights("dqn.h5", overwrite=True)
        	print("TIMESTEP: "+ str(timestep) + ", EPSILON: " + str(epsilon) + ", ACTION: " + str(action) + ", REWARD: " + str(reward) + ", Loss: " + str(loss))
        	loss = 0
        elif not alive:
        	print("EPISODE: " + str(episode) + ", SCORE: " + str(score))
        	sfile.write(str(score)+"\n")
        	episode += 1

if __name__ == "__main__":
	model = build_network()
	train_network(model,sys.argv[1])