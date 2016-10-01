import numpy as np


def h(weights, X):
    """ Calculate the hypothesis value.
    
    Parameters:
    weights -- an nx1 column vector, where n is the number of features/variables plus 1 for the bias/intercept
    X -- an mxn matrix, where m is the number of training examples, plus a column of 1's for the bias
    
    Return:
    An mx1 column vector containing the hypothesis values for each training example
    """

    return np.dot(X, weights)


def calculate_cost(weights, X, Y):
    """Calculate total cost across all samples and features for a given set of weights
    
    Parameters:
    weights -- nx1 column vector, where n is the number of features/variables plus 1 for the bias/intercept
    X -- mxn matrix, where m is the number of training examples, plus a column of 1's for the bias
    Y -- mx1 column vector containing the labels/targets for each training example
    
    Returns:
    residuals -- mx1 array of residuals for each training example
    total_cost -- 1x1 float value representing the average squared error across the entire dataset for a given set of weights
    """
    m = Y.shape[0]  # Number of training examples. Equivalent to X.shape[0]
    residuals = h(weights, X) - Y  # mx1 column vector containing residual for each training example
    squared_error = np.dot(residuals.T, residuals)  # 1x1 containing the total squared error
    total_cost = float(1)/(2*m) * squared_error  # 1x1 containing average cost value over all training examples
    
    return residuals, total_cost


def batch_gradient_descent(weights, X, Y, iterations = 1000, alpha = 1e-6, verbose = True):
    """Update weight values using gradient descent
    
    Parameters:
    weights -- nx1 column vector, where n is the number of features/variables plus 1 for the bias/intercept
    X -- mxn matrix, where m is the number of training examples, plus a column of 1's for the bias
    Y -- mx1 column vector containing the labels/targets for each training example
    iterations -- number of training iterations to perform
    alpha -- learning rate. Step size multiplier for each weight adjustment
    verbose -- print the current iteration number and current cost value
    
    Returns:
    cost_history -- numpy array containing the cost values for each iteration
    theta -- vector of final weights after all training iterations
    """
    
    theta = weights
    m = Y.shape[0]  # Number of training examples. Equivalent to X.shape[0]
    cost_history = np.zeros(iterations)  # Initialize array of cost history values with 0's

    for i in xrange(iterations):
        residuals, cost = calculate_cost(theta, X, Y)
        gradient = (float(1)/m) * np.dot(residuals.T, X).T  #nx1 column vector containing current gradient of each variable
        theta -= (alpha * gradient)  #nx1 column vector containing updated all updated weight values
        
        # Store the cost for this iteration
        cost_history[i] = cost
        
        if verbose:
            print "Iteration: %d | Cost: %f" % (i+1, cost)

    return cost_history, theta