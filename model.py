import tensorflow as tf 
import csv
import pandas as pd 
import numpy as np 
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras import initializers

train_data = pd.read_csv('train.csv', delimiter=',')
train_data = train_data[1:]
for column in train_data.columns:
	train_data[column] = train_data[column].convert_objects(convert_numeric=True)
X = train_data.iloc[:,0:len(train_data.columns)-1]
Y = train_data.iloc[:,len(train_data.columns)-1]

model = Sequential()
model.add(Dense(20,kernel_initializer=initializers.random_normal(stddev=0.01),input_dim=6))
model.add(Activation('relu'))
model.add(Dense(30,kernel_initializer=initializers.random_normal(stddev=0.01)))
model.add(Activation('relu'))
model.add(Dense(20,kernel_initializer=initializers.random_normal(stddev=0.01)))
model.add(Activation('relu'))
model.add(Dense(1,kernel_initializer=initializers.random_normal(stddev=0.01)))
model.compile(loss='mse',optimizer='adam')
print("Model Ready")

model.fit(X,Y,batch_size=1000,nb_epoch=100)
model.save('nn.h5')