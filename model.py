import tensorflow as tf 
import csv
import numpy as np 
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras import initializers

train_data = np.genfromtxt("train.csv", dtype=float, delimiter=',', names=True)

train_X = np.matrix([train_data["x_dist_pipe"],train_data["y_dist_upipe"],train_data["y_dist_lpipe"],train_data["playerVelY"]])
train_y = np.matrix(train_data["playerFlapped"])
train_X = train_X.transpose()
train_y = train_y.transpose()

model = Sequential()
model.add(Dense(20,kernel_initializer=initializers.random_normal(stddev=0.01),input_dim=4))
model.add(Activation('relu'))
model.add(Dense(30,kernel_initializer=initializers.random_normal(stddev=0.01)))
model.add(Activation('relu'))
model.add(Dense(20,kernel_initializer=initializers.random_normal(stddev=0.01)))
model.add(Activation('softmax'))
model.add(Dense(1,kernel_initializer=initializers.random_normal(stddev=0.01)))
model.compile(loss='mse',optimizer='adam')
print("Model Ready")

model.fit(train_X,train_y,batch_size=1000,epochs=100)
model.save('nn.h5')