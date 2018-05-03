import sys
import os
from flappy_dqn import State
import random
import numpy as np
from collections import deque

import tensorflow as tf
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten
from keras.layers.convolutional import Conv2D
from keras.optimizers import SGD , Adam

import skimage as skimage
from skimage import transform, color, exposure
from skimage.transform import rotate

ACTIONS = 2 # number of valid actions
DISCOUNT_FACTOR = 0.95 # decay rate of past observations
OBSERVE = 5000 # timesteps to observe before training
EXPLORE = 3000000 # frames over which to anneal epsilon
FINAL_EPSILON = 0.0001 # final value of epsilon
INITIAL_EPSILON = 0.1 # starting value of epsilon
REPLAY_MEMORY = 5000 # number of previous transitions to remember
TRAIN = True

def network():
    model = Sequential()

    model.add(Conv2D(32, (6, 6), padding='valid', strides=(4, 4), input_shape=(60,60,4), activation='relu'))  
    model.add(Activation('relu'))
    model.add(Conv2D(64, (4, 4), padding='valid', strides=(2, 2)))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3), padding='valid', strides=(1, 1)))
    model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dense(2))

    adam = Adam(lr=1e-4)
    model.compile(loss='mse',optimizer=adam)

    return model

def train(model):
    global OBSERVE
    sfile = open("scores_dqn.txt","a+")
    episode = 1
    game = State()
    replay = deque()

    image, score, reward, alive = game.next(0)

    image = skimage.color.rgb2gray(image)
    image = skimage.transform.resize(image,(60,60))
    image = skimage.exposure.rescale_intensity(image,out_range=(0,255))
    image = image / 255.0
    input_image = np.stack((image, image, image, image), axis=2)
    input_image = input_image.reshape(1, input_image.shape[0], input_image.shape[1], input_image.shape[2])

    if TRAIN:
        epsilon = INITIAL_EPSILON
        if os.path.exists("dqn.h5"):
            model.load_weights("dqn.h5")
            adam = Adam(lr=1e-4)
            model.compile(loss='mse',optimizer=adam)    
    
    timestep = 0
    while (True):
        loss = 0
        
        if random.random() <= epsilon:
            print("----------Random Action----------")
            action = random.randrange(ACTIONS)
        else:
            q = model.predict(input_image)       
            action = np.argmax(q)
        
        if epsilon > FINAL_EPSILON and timestep > OBSERVE:
            epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE

        image1, score, reward, alive = game.next(action)

        image1 = skimage.color.rgb2gray(image1)
        image1 = skimage.transform.resize(image1,(60,60))
        image1 = skimage.exposure.rescale_intensity(image1, out_range=(0, 255))
        image1 = image1 / 255.0
        image1 = image1.reshape(1, image1.shape[0], image1.shape[1], 1) #1x80x80x1
        input_image1 = np.append(image1, input_image[:, :, :, :3], axis=3)

        replay.append((input_image, action, reward, input_image1, alive))
        if len(replay) > REPLAY_MEMORY:
            replay.popleft()

        if not alive:
            print("EPISODE: " + str(episode) + ", SCORE: " + str(score) + "-------------------------")
            sfile.write(str(score)+"\n")
            episode += 1

        if timestep > OBSERVE:
            minibatch = random.sample(replay, 32)
            s, a, r, s1, alive = zip(*minibatch)
            s = np.concatenate(s)
            s1 = np.concatenate(s1)
            targets = model.predict(s)
            Q_sar = model.predict(s1)
            targets[range(32), a] = r + DISCOUNT_FACTOR*np.max(model.predict(s1), axis=1)*alive
            loss += model.train_on_batch(s, targets)

        input_image = input_image1
        timestep = timestep + 1

        if timestep % 500 == 0:
            model.save_weights("dqn.h5", overwrite=True)
        
        print("TIMESTEP: "+ str(timestep) + ", EPSILON: " + str(epsilon) + ", ACTION: " + str(action) + ", REWARD: " + str(reward) + ", Loss: " + str(loss))

def main():
    model = network()
    train(model)

if __name__ == "__main__":
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    from keras import backend as K
    K.set_session(sess)
    main()