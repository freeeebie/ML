# -*- coding: utf-8 -*-

# Machine Learning Online Class - Exercise 2: Logistic Regression

import numpy as np
import matplotlib.pyplot as plt
from pylab import *
from mpl_toolkits.mplot3d import Axes3D

def plotData(X,y):
	pos = np.where(y == 1)
	neg = np.where(y == 0)
	print(pos)
	pos_list = pos[0].tolist()
	neg_list = neg[0].tolist()
	print(pos_list)
	# pos_list2 = 
	X_pos1 = map(lambda x:X[x][0], pos_list)
	X_pos2 = map(lambda x:X[x][1], pos_list)

	X_neg = map(lambda x:[X[x][0], X[x][1]], neg_list)
	print(X_pos2)
	# print(X_pos[:, 0])

	plt.plot(X_pos1,X_pos2, 'k+',linewidth=2,     markersize = 7);
#	plt.plot(X_neg, marker = '.') #, 'MarkerFaceColor', 'y',     'MarkerSize', 7);
	plt.show()
	pass
	# plt.plot(X, y, 'rx', markersize = 10); # Plot the data
	# plt.ylabel('Profit in $10,000s'); # Set the y−axis label
	# plt.xlabel('Population of City in 10,000s'); # Set the x−axis label
	# plt.ion()
	# plt.show()
	# pos = find(y==1); neg = find(y == 0);
	# % Plot Examples
	# plot(X(pos, 1), X(pos, 2), 'k+','LineWidth', 2, ...
	#      'MarkerSize', 7);
	# plot(X(neg, 1), X(neg, 2), 'ko', 'MarkerFaceColor', 'y', ...
	#      'MarkerSize', 7);

	# %% Load Data
	# %  The first two columns contains the exam scores and the third column
	# %  contains the label.

data = np.loadtxt('ex2data1.txt', dtype=float, delimiter=',')
print(data)
X = data[:,[0,1]]
y = data[:,2]

# %% ==================== Part 1: Plotting ====================
# %  We start the exercise by first plotting the data to understand the
# %  the problem we are working with.

print(['Plotting data with + indicating (y = 1) examples and o indicating (y = 0) examples.'])

plotData(X, y);
print(X)

# % Put some labels
# hold on;
# % Labels and Legend
# xlabel('Exam 1 score')
# ylabel('Exam 2 score')

# % Specified in plot order
# legend('Admitted', 'Not admitted')
# hold off;

# fprintf('\nProgram paused. Press enter to continue.\n');
# pause;


# %% ============ Part 2: Compute Cost and Gradient ============
# %  In this part of the exercise, you will implement the cost and gradient
# %  for logistic regression. You neeed to complete the code in
# %  costFunction.m

# %  Setup the data matrix appropriately, and add ones for the intercept term
# [m, n] = size(X);

# % Add intercept term to x and X_test
# X = [ones(m, 1) X]

# % Initialize fitting parameters
# initial_theta = zeros(n + 1, 1)

# % Compute and display initial cost and gradient
# [cost, grad] = costFunction(initial_theta, X, y);

# fprintf('Cost at initial theta (zeros): %f\n', cost);
# fprintf('Gradient at initial theta (zeros): \n');
# fprintf(' %f \n', grad);

# fprintf('\nProgram paused. Press enter to continue.\n');
# pause;


# %% ============= Part 3: Optimizing using fminunc  =============
# %  In this exercise, you will use a built-in function (fminunc) to find the
# %  optimal parameters theta.

# %  Set options for fminunc
# options = optimset('GradObj', 'on', 'MaxIter', 400);

# %  Run fminunc to obtain the optimal theta
# %  This function will return theta and the cost
# [theta, cost] = ...
# 	fminunc(@(t)(costFunction(t, X, y)), initial_theta, options);

# % Print theta to screen
# fprintf('Cost at theta found by fminunc: %f\n', cost);
# fprintf('theta: \n');
# fprintf(' %f \n', theta);

# % Plot Boundary
# plotDecisionBoundary(theta, X, y);

# % Put some labels
# hold on;
# % Labels and Legend
# xlabel('Exam 1 score')
# ylabel('Exam 2 score')

# % Specified in plot order
# legend('Admitted', 'Not admitted')
# hold off;

# fprintf('\nProgram paused. Press enter to continue.\n');
# pause;

# %% ============== Part 4: Predict and Accuracies ==============
# %  After learning the parameters, you'll like to use it to predict the outcomes
# %  on unseen data. In this part, you will use the logistic regression model
# %  to predict the probability that a student with score 45 on exam 1 and
# %  score 85 on exam 2 will be admitted.
# %
# %  Furthermore, you will compute the training and test set accuracies of
# %  our model.
# %
# %  Your task is to complete the code in predict.m

# %  Predict probability for a student with score 45 on exam 1
# %  and score 85 on exam 2

# prob = sigmoid([1 45 85] * theta);
# fprintf(['For a student with scores 45 and 85, we predict an admission ' ...
#          'probability of %f\n\n'], prob);

# % Compute accuracy on our training set
# p = predict(theta, X);

# fprintf('Train Accuracy: %f\n', mean(double(p == y)) * 100);

# fprintf('\nProgram paused. Press enter to continue.\n');
# pause;
