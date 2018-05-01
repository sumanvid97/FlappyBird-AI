import numpy as np
from keras.models import Sequential
from keras.layers import Dense

dataset = np.loadtxt("../train.csv", delimiter=",")
# split into input (X) and output (Y) variables
X = dataset[:,0:3]
y = dataset[:,3]

model = Sequential()
model.add(Dense(20, input_dim=3, activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam')
model.fit(X, y, epochs=50, batch_size=1000)
model.save('nn.h5')