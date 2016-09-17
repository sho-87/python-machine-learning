import numpy as np
from sklearn import datasets

# Linear regression
def linear_regression_data(samples = 5000, predictors = 50):
    regression_features, regression_labels = datasets.make_regression(samples, predictors)
    np.savetxt('linear_regression/regression_features.csv', regression_features, delimiter=',')
    np.savetxt('linear_regression/regression_labels.csv', regression_labels, delimiter=',')

# Logistic regression
def logistic_regression_data(samples = 5000):
    logistic_features, logistic_labels = datasets.make_hastie_10_2(samples)
    np.savetxt('logistic_regression/logistic_features.csv', logistic_features, delimiter=',')
    np.savetxt('logistic_regression/logistic_labels.csv', logistic_labels, delimiter=',')
    
# Neural network AND gate
def NN_AND_data(samples = 5000):
    # Generate random inputs for AND gate
    input_a = np.random.choice([0, 1], samples)
    input_b = np.random.choice([0, 1], samples)
    
    # Create feature matrix from the random input values above
    features = np.column_stack([input_a, input_b])
    
    # Create the labels
    labels = np.logical_and(input_a == 1, input_b == 1).astype(int)
    
    # Save data
    np.savetxt('logic_gates/NN_AND_features.csv', features, delimiter=',')
    np.savetxt('logic_gates/NN_AND_labels.csv', labels, delimiter=',')

# Neural network OR gate
def NN_OR_data(samples = 5000):
    # Generate random inputs for OR gate
    input_a = np.random.choice([0, 1], samples)
    input_b = np.random.choice([0, 1], samples)
    
    # Create feature matrix from the random input values above
    features = np.column_stack([input_a, input_b])
    
    # Create the labels
    labels = np.logical_or(input_a == 1, input_b == 1).astype(int)
    
    # Save data
    np.savetxt('logic_gates/NN_OR_features.csv', features, delimiter=',')
    np.savetxt('logic_gates/NN_OR_labels.csv', labels, delimiter=',')
    
# Neural network XOR gate
def NN_XOR_data(samples = 5000):
    # Generate random inputs for XOR gate
    input_a = np.random.choice([0, 1], samples)
    input_b = np.random.choice([0, 1], samples)
    
    # Create feature matrix from the random input values above
    features = np.column_stack([input_a, input_b])
    
    # Create the labels
    labels = np.logical_xor(input_a == 1, input_b == 1).astype(int)
    
    # Save data
    np.savetxt('logic_gates/NN_XOR_features.csv', features, delimiter=',')
    np.savetxt('logic_gates/NN_XOR_labels.csv', labels, delimiter=',')