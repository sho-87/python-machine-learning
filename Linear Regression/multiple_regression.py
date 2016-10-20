# Basic OLS regression using base python (and numpy)

import numpy as np
import matplotlib.pyplot as plt

# Load data
dataset = np.genfromtxt('../data/regression_heart.csv', delimiter=",")

x = dataset[:,1:]
y = dataset[:,0]
y = np.reshape(y, (y.shape[0],1))  # Reshape to a column vector

# Scale (standardize) data for smoother gradient descent
x = (x - np.mean(x, axis=0)) / np.std(x, axis=0)
x = np.insert(x, 0, 1, axis=1)  # Add 1's for bias

# Learning parameters
alpha = 0.01  # Learning rate
iterations = 1000

# Notes
# Cost function for linear regression: 1/2m(sum((theta0 + theta1(x) - y) ^ 2))
# Partial derivative wrt theta0: 1/m(sum(theta0 + theta1 - y))
# Partial derivative wrt theta1: 1/m(sum(theta0 + theta1 - y)) * x
# Parameter update: theta = theta - alpha(partial derivative)

# Training
theta = np.ones((x.shape[1],1))  # Initial weights
m = y.shape[0]  # Number of training examples. Equivalent to X.shape[0]
cost_history = np.zeros(iterations)  # Initialize array of cost history values

for i in xrange(iterations):  # Batch gradient descent
    residuals = np.dot(x, theta) - y
    squared_error = np.dot(residuals.T, residuals)
    cost = float(1)/(2*m) * squared_error  # Quadratic loss
    
    gradient = (float(1)/m) * np.dot(residuals.T, x).T  # Calculate derivative
    theta -= (alpha * gradient)  # Update weights
    
    cost_history[i] = cost  # Store the cost for this iteration
    
    if (i+1) % 100 == 0:
        print "Iteration: %d | Cost: %f" % (i+1, cost)

# Plot training curve
plt.plot(range(1, len(cost_history)+1), cost_history)
plt.grid(True)
plt.xlim(1, len(cost_history))
plt.ylim(0, max(cost_history))
plt.title("Training Curve")
plt.xlabel("Iteration #")
plt.ylabel("Cost")
