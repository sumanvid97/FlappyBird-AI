import csv
fo = open('ntrain.csv','r')
of = open('train.csv','w')
for i in range(21001):
	line = fo.readline()
	of.write(line)
of.close()
fo.close()