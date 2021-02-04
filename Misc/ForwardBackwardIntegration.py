import future, sys, os, datetime, argparse
# print(os.path.dirname(sys.executable))
import torch
import numpy as np
import matplotlib
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

matplotlib.rcParams["figure.figsize"] = [10, 10]

'''
d sin(x) / dx = cos(x)
d^2 sin(x) / dx^2 = - sin(x)
'''

x = list(np.linspace(0, 1.2*np.pi, 50))
step_size = x[-1]/len(x)

plt.title('Ground Truth')
plt.plot(x, np.sin(x), label='sin(x)')
plt.plot(x, np.cos(x), label='d sin(x) = cos(x)')
plt.plot(x, -np.sin(x), ls='--', label='d^2 sin(x) = -sin(x)')
plt.legend()
plt.grid()
plt.show()

'''Forward Integration'''

x_ = [0]+list(np.cumsum(np.cos(x)*step_size))

if False:
	plt.plot(np.sin(x), label='True sin(x)')
	plt.plot(x_, label='Integrated sin(x)')
	plt.legend()
	plt.grid()
	plt.show()

####################################################################################################################
####################################################################################################################
####################################################################################################################

'''Backward Integration'''
x_ = [np.sin(x[-1])]
dx_ = [np.cos(x[-1])]

'''Going forward in Array by concatenating to the end in the Backward integration'''
for t in reversed(x):
	'''Take last entry in array indexed by t and stick the prediction to the end'''
	x_ 	= x_ + [x_[-1] - np.cos(t)*step_size]
	dx_	= dx_ + [dx_[-1] -(-np.sin(t)*step_size)]

x_.reverse() 	# in_place reverse
dx_.reverse()	# in_place reverse

plt.title('Concatenation to Back of Backward Integration')
plt.plot(x, np.sin(x), c='r', label='True sin(x)')
plt.plot(x, np.cos(x), c='b', label='True dsin(x)')
plt.plot([0] + x, x_, ls='--', c='r', label='Reverse Integrated sin(x)')
plt.plot([0] + x, dx_, ls='--', c='b', label='Reverse Integrated d sin(x)')
plt.legend()
plt.grid()
plt.show()

####################################################################################################################
####################################################################################################################
####################################################################################################################

'''Init initial conditions (acutally final condition since we integrate backward)'''
x_ = [np.sin(x[-1])]
dx_ = [np.cos(x[-1])]
for t in reversed(x):
	'''Take the first entry in array and stick the step to the front'''
	'''Moves literally along the time axis from right to left through the array'''
	x_ = [x_[-0] - np.cos(t) * step_size] + x_
	dx_ = [dx_[0] - (-np.sin(t) * step_size)] + dx_

plt.title('Concatenation to Front of Backward Integration')
plt.plot(x, np.sin(x), c='r', label='True sin(x)')
plt.plot(x, np.cos(x), c='b', label='True dsin(x)')
plt.plot([0]+x, x_, ls='--', c='r', label='Reverse Integrated sin(x)')
plt.plot([0]+x, dx_, ls='--', c='b', label='Reverse Integrated d sin(x)')
plt.legend()
plt.grid()
plt.show()