import numpy as np

def scale_standardize(features):
    """Standardize the input features.
    
    Parameters:
    features -- mxn matrix, where m is the number of training examples, and n is the number of features/variables. No bias column should be included
    
    Returns:
    x -- feature matrix with standardized values
    """
    
    x = (features - np.mean(features, axis=0)) / np.std(features, axis=0)
    x = np.insert(x, 0, 1, axis=1)  # Add 1's for bias

    return x