import numpy as np
import matplotlib.pyplot as plt

dataset = np.loadtxt('scores.txt', delimiter=',')
# split into input (X) and output (Y) variables
x = dataset[:,0]
y = dataset[:,1] 
# plotting points as a scatter plot
plt.scatter(x, y, s=5)
 
# x-axis label
plt.xlabel('Iterations')
# frequency label
plt.ylabel('Scores')
# plot title
plt.title('Score')
# showing legend
plt.legend()
 
# function to show the plot
plt.savefig('plot.png')
plt.show()

