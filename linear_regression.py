import numpy as np

# Generate random inputs for AND gate
num_samples = 10
input_a = np.random.choice([0, 1], num_samples)
input_b = np.random.choice([0, 1], num_samples)

# Create feature matrix from the random input values above
feature_matrix = np.column_stack([input_a, input_b])

# Create the labels
labels = np.logical_and(input_a == 1, input_b == 1).astype(int)

