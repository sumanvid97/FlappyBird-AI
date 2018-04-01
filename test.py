import tensorflow as tf 
import csv
import pandas as pd 
import numpy as np 
from keras.models import load_model

model = load_model('nn.h5')
train_data = np.genfromtxt("train.csv", dtype=float, delimiter=',', names=True)

train_X = np.matrix([train_data["x_dist_pipe"],train_data["y_dist_upipe"],train_data["y_dist_lpipe"],train_data["playerVelY"]])
train_y = np.matrix(train_data["playerFlapped"])
train_X = train_X.transpose()
train_y = train_y.transpose()
train_y = train_y.tolist()

y = np.zeros(len(train_y))
y = model.predict_classes(train_X)

count = 0
count1 = 0
count2 = 0

for i in range(len(y)):
	if y[i][0] == train_y[i][0]:
		count += 1
	if y[i][0]:
		count1 += 1
	if not y[i][0]:
		count2 += 1
print('Training Accuracy', float(count)/float(len(y)))
print(count)
print(count1)
print(count2)
print(len(y))

