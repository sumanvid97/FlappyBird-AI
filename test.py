import numpy as np 
from keras.models import load_model

model = load_model('nn.h5')
dataset = np.loadtxt("../test.csv", delimiter=",")
# dataset = np.loadtxt("record.csv", delimiter=",")
# split into input (X) and output (Y) variables
test_X = dataset[:,0:3]
test_y = dataset[:,3]
y = model.predict(test_X)

count = 0
count1 = 0

for i in range(len(y)):
	if int(round(y[i])) == test_y[i]:
		count += 1
	if int(round(y[i])):
		count1 += 1
print('Testing Accuracy', float(count)/float(len(y)))
print(count)
print(count1)
print(len(y))