"""
HW_1.py
AM 230 - Numerical Optimization
Programmer: Bryan Garcia
Date: 26 January 2020

Description: Illustrates properties of "Steepest Descent" algorithm.
"""

import matplotlib
matplotlib.use("tkagg")

import numpy as np
from scipy.optimize import line_search
import matplotlib.pyplot as plt

"""
Cost Function, Gradient, and Hessian
"""

def cost_fun(x, c = 1.0):
	return (c * x[0] - 2) ** 4 + x[1] ** 2 * (c * x[0] - 2) ** 2 + (x[1] + 1) ** 2 

def grad(x, c = 1.0):
	return np.array([(4 * c) * (c * x[0] - 2) ** 3 + ( (2 * x[1]) ** 2 ) * (c * x[0] - 2), \
					(2 * x[1]) * (c * x[0] - 2)**2 + 2 * (x[1] + 1)])

def hessian(x, c = 1.0):
	return np.array([ 
		np.array([(12 * c ** 2) * (c * x[0] - 2) ** 2 + (2 * c ** 2) * (x[1] ** 2), 4*c*x[1]*(c*x[0]-2)]), \
		np.array([(4 * c ** 2) * (x[0] * x[1]) - (8 * c) * x[1], 2 * (c * x[0] - 2)**2 + 2]) \
	])

# Max iteration count and tolerance (eps) settings
max_iter = 1000
eps = 1e-8

error_list = []

x_initial = np.random.rand(2)


# Iterating over c = 1 and c = 10 values separately...
for c in [1, 10]:

	# Global minimizer 
	cost_min = np.array([2 / c, -1])

	# Generating random values for initial point and computing initial search direction
	x_vals = np.zeros((max_iter, 2))
	x_vals[0] = x_initial
	p_curr = -grad(x_vals[0], c)

	error = np.zeros(max_iter)
	error[0] = np.abs(cost_fun(x_vals[0], c) - cost_fun(cost_min, c))

	num_iter = 0

	# Beginning Steepest Descent...
	for i in range(max_iter + 1):
		if np.linalg.norm(p_curr) > eps:

			# Compute step-size value satisfying strong Wolfe conditions
			a_step = line_search(cost_fun, grad, x_vals[i], p_curr, args = [c])[0]


			print("=" * 80)
			print("Current iteration: {}".format(i))
			print("(X1, X2)_{}: ({}, {})".format(i, x_vals[i][0], x_vals[i][1]))
			print("cost_fun({}) = {}".format(x_vals[i], cost_fun(x_vals[i], c)))
			print("Error = {}".format(error[i]))
			print("Eigenvalues: {}".format(np.linalg.eig(hessian(x_vals[i], c))[0]))
			print("=" * 80 + "\n")


			# Store next iteration point and update search direction
			x_vals[i+1] = x_vals[i] + a_step * p_curr
			error[i+1] = np.abs(cost_fun(x_vals[i+1], c) - cost_fun(cost_min, c))
			p_curr = -grad(x_vals[i + 1], c)

		else:
			num_iter = i
			error_list.append((num_iter, error, c))
			break

# Plot data we gathered...
for num_iter, error, c in error_list:
	plt.plot([i for i in range(num_iter)], [np.log(error[i]) for i in range(num_iter)], label = r'$C =$ {}'.format(c))
plt.legend()
plt.title(r'SD with $x_0$ = ({}, {})'.format(x_initial[0], x_initial[1]))
plt.xlabel(r'Number of iterations $k$', fontsize = 15)
plt.ylabel(r'$|\log(f(x_k) - f(x^{*}))|$', fontsize = 15)
plt.grid(True)
plt.show()






















