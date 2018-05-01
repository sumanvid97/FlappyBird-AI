import csv

input1 = open('../ntrain.csv','r')
train = open('../train.csv','w')
line = input1.readline()
while len(line)>0:
	line = input1.readline()
	train.write(line)

input2 = open('../ntest.csv','r')
test = open('../test.csv','w')
line = input2.readline()
while len(line)>0:
	line = input2.readline()
	test.write(line)

input1.close()
input2.close()
train.close()
test.close()