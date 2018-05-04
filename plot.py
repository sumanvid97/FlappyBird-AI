import numpy as np
import sys
import matplotlib.pyplot as plt

y = np.loadtxt(sys.argv[1], delimiter=',')
x = range(len(y)) 
# plotting points as a scatter plot
plt.scatter(x, y, s=1)
 
# x-axis label
plt.xlabel('Iterations')
# frequency label
plt.ylabel('Scores')
# plot title
plt.title('Score')
# showing legend
plt.legend()
 
# function to show the plot
plt.savefig('plot'+sys.argv[1].split(".")[0]+'.png')
plt.show()

