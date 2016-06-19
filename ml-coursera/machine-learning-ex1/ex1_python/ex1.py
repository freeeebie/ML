# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from pylab import *
from mpl_toolkits.mplot3d import Axes3D

def warmUpExercise():
	A = np.eye(5)
	print(A)

def plotData(X,y):
	plt.plot(X, y, 'rx', markersize = 10); # Plot the data
	plt.ylabel('Profit in $10,000s'); # Set the y−axis label
	plt.xlabel('Population of City in 10,000s'); # Set the x−axis label
	plt.ion()
	plt.show()

def computeCost(X, y, theta):
	m = len(y); # number of training examples

	J = 0;

# % Instructions: Compute the cost of a particular choice of theta
# %               You should set J to the cost.

	h_theta = np.dot((theta.transpose()) , (X.transpose()) )
	J = (1./(2*m)) * np.sum(      np.multiply((h_theta - y),(h_theta - y)))
	print(J)

def gradientDescent(X, y, theta, alpha, num_iters): 
	# % Initialize some useful values
	m = len(y); #% number of training examples
	J_history = [] #* num_iters # = np.zeros((num_iters, 1))

	for i in range(num_iters):
		h_theta = np.dot((theta.transpose()) , (X.transpose()) )
		# print(h_theta)
		theta = theta - alpha * (1. / m) * np.sum   ( np.multiply((h_theta - y),(X[:,1])));
		# print(theta)	
		J_history.append(computeCost(X,y,theta))
	print(theta)
	return theta

print('Running warmUpExercise ... \n');
print('5x5 Identity Matrix: \n');
warmUpExercise()

print('Program paused. Press enter to continue.\n');

print('Plotting Data ...\n')
data = np.loadtxt('ex1data1.txt', dtype=float, delimiter=',');
X = data[:,0]; 
y = data[:,1];
m = len(y); # number of training examples

plotData(X, y);

print('Program paused. Press enter to continue.\n');


# =================== Part 3: Gradient descent ===================
print('Running Gradient Descent ...\n')

temp = np.ones((m, 1))
print(temp.tolist())
print(data[:,0])
X = np.c_[temp, data[:,0]]  #% Add a column of ones to x
theta = np.zeros((2, 1)); # initialize fitting parameters

print(X)
# Some gradient descent settings
iterations = 1500;
alpha = 0.01;

# compute and display initial cost
computeCost(X, y, theta)

# run gradient descent
theta = gradientDescent(X, y, theta, alpha, iterations);

print(theta)

plt.plot(X[:,1], np.dot(X,theta), '-')
plt.draw()
plt.pause(10.001)

# Predict values for population sizes of 35,000 and 70,000
predict1 = np.dot([1, 3.5] , theta );
print('For population = 35,000, we predict a profit of %f\n', (predict1*10000)[0])
predict2 = np.dot([1, 7] , theta)
print('For population = 70,000, we predict a profit of %f\n',  (predict2*10000)[0])


# ============= Part 4: Visualizing J(theta_0, theta_1) =============
print('Visualizing J(theta_0, theta_1) ...')

# Grid over which we will calculate J
theta0_vals = linspace(-10, 10, 100);
theta1_vals = linspace(-1, 4, 100);

# initialize J_vals to a matrix of 0's
J_vals = zeros((len(theta0_vals), len(theta1_vals)));

# Fill out J_vals
for i in range (len(theta0_vals)):
	for j in range (len(theta1_vals)):
		t = np.array([[theta0_vals[i]], [theta1_vals[j]]])
		J_vals[i][j] = computeCost(X, y, t);


# Because of the way meshgrids work in the surf command, we need to
# transpose J_vals before calling surf, or else the axes will be flipped
J_vals = J_vals.transpose();
# Surface plot
# figure;
# surf(theta0_vals, theta1_vals, J_vals)
# xlabel('\theta_0'); ylabel('\theta_1');

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
x = y = np.arange(-3.0, 3.0, 0.05)
X, Y = np.meshgrid(x, y)
# zs = np.array([fun(x,y) for x,y in zip(np.ravel(X), np.ravel(Y))])
# Z = zs.reshape(X.shape)

ax.plot_surface(theta0_vals, theta1_vals, J_vals)

ax.set_xlabel('theta_0')
ax.set_ylabel('theta_1')
ax.set_zlabel('Z Label')
plt.show()
# % Contour plot
# figure;
# % Plot J_vals as 15 contours spaced logarithmically between 0.01 and 100
# contour(theta0_vals, theta1_vals, J_vals, logspace(-2, 3, 20))
# xlabel('\theta_0'); ylabel('\theta_1');
# hold on;
# plot(theta(1), theta(2), 'rx', 'MarkerSize', 10, 'LineWidth', 2);




