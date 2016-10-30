# XOR gate. ANN with 1 hidden layer. Noise added to test set

import numpy as np
import theano
import theano.tensor as T
import matplotlib.pyplot as plt

# Set inputs and correct output values
inputs = [[0,0], [1,1], [0,1], [1,0]]
outputs = [0, 0, 1, 1]

# Set training parameters
alpha = 0.1  # Learning rate
training_iterations = 50000
hidden_layer_nodes = 3

 # Define tensors
x = T.matrix("x")
y = T.vector("y")
b1 = theano.shared(value=1.0, name='b1')
b2 = theano.shared(value=1.0, name='b2')

# Set random seed
rng = np.random.RandomState(2345)

# Initialize weights
w1_array = np.asarray(
    rng.uniform(low=-1, high=1, size=(2, hidden_layer_nodes)),
    dtype=theano.config.floatX)  # Force type to 32bit float for GPU
w1 = theano.shared(value=w1_array, name='w1', borrow=True)

w2_array = np.asarray(
    rng.uniform(low=-1, high=1, size=(hidden_layer_nodes, 1)),
    dtype=theano.config.floatX)  # Force type to 32bit float for GPU
w2 = theano.shared(value=w2_array, name='w2', borrow=True)

# Theano symbolic expressions
a1 = T.nnet.sigmoid(T.dot(x, w1) + b1)  # Input -> Hidden
a2 = T.nnet.sigmoid(T.dot(a1, w2) + b2)  # Hidden -> Output
hypothesis = T.flatten(a2)  # This needs to be flattened so
                            # hypothesis (matrix) and
                            # y (vector) have the same shape

# cost = T.sum((y - hypothesis) ** 2)  # Quadratic/squared error loss
# cost = -(y*T.log(hypothesis) + (1-y)*T.log(1-hypothesis)).sum()  # Manual CE
# cost = T.nnet.categorical_crossentropy(hypothesis, y)  # Categorical CE
cost = T.nnet.binary_crossentropy(hypothesis, y).mean()  # Binary CE

updates_rules = [
    (w1, w1 - alpha * T.grad(cost, wrt=w1)),
    (w2, w2 - alpha * T.grad(cost, wrt=w2)),
    (b1, b1 - alpha * T.grad(cost, wrt=b1)),
    (b2, b2 - alpha * T.grad(cost, wrt=b2))
    ]

# Theano compiled functions
train = theano.function(inputs=[x, y], outputs=[hypothesis, cost],
                        updates=updates_rules)
predict = theano.function(inputs=[x], outputs=[hypothesis])

# Training
cost_history = []

for i in range(training_iterations):
    if (i+1) % 5000 == 0:
        print "Iteration #{}: ".format(i+1)
        print "Cost: {}".format(cost)
    h, cost = train(inputs, outputs)
    cost_history.append(cost)

# Plot training curve
plt.figure(0)
plt.plot(range(1, len(cost_history)+1), cost_history)
plt.grid(True)
plt.xlim(1, len(cost_history))
plt.ylim(0, max(cost_history))
plt.title("Training Curve")
plt.xlabel("Iteration #")
plt.ylabel("Cost")

# Predictions on noisy test data
num_reps = 2000  # Number of times to repeat the test set
labels = [0, 0, 1, 1] * num_reps
raw_input = [[0,0], [1,1], [0,1], [1,0]] * num_reps
raw_input = np.asarray(raw_input, dtype='float32')  # Cast to float32

# Wide range
sd_list = np.arange(0.05, 2.05, 0.05)  #  Sequence of standard deviations
accuracy_history = []

for sd in sd_list:
    # Add random noise to the input data (sampled from normal distribution)
    spread = np.random.normal(0, sd, (len(raw_input), 2)).astype('float32')
    pred_data = np.add(raw_input, spread)
    
    predictions = predict(pred_data)[0]
    predictions[predictions >= 0.5] = 1
    predictions[predictions < 0.5] = 0
    
    # Calculate % accuracy for this SD set
    cur_acc = float(np.sum(predictions == labels))/len(predictions) * 100
    accuracy_history.append(cur_acc)
    # print "Standard deviation: {}".format(sd)
    # print "Accuracy: {}%".format(cur_acc, prec=1)

# Plot accuracy over standard deviations (wide range)
plt.figure(1)
plt.plot(sd_list, accuracy_history)
plt.axhline(y=50, color='r', linestyle='--')
plt.grid(True)
plt.xlim(min(sd_list), max(sd_list))
plt.ylim(0, max(accuracy_history)+10)
plt.title("Test Set Accuracy")
plt.xlabel("Noise (Standard Deviations)")
plt.ylabel("Classification Accuracy (%)")

# Narrow range (higher resolution of the 0-0.5 SD range)
sd_list = np.arange(0.05, .55, 0.05) #  Sequence of standard deviations
accuracy_history = []

for sd in sd_list:
    # Add random noise to the input data (sampled from normal distribution)
    spread = np.random.normal(0, sd, (len(raw_input), 2)).astype('float32')
    pred_data = np.add(raw_input, spread)
    
    predictions = predict(pred_data)[0]
    predictions[predictions >= 0.5] = 1
    predictions[predictions < 0.5] = 0
    
    # Calculate % accuracy for this SD set
    cur_acc = float(np.sum(predictions == labels))/len(predictions) * 100
    accuracy_history.append(cur_acc)
    # print "Standard deviation: {}".format(sd)
    # print "Accuracy: {}%".format(cur_acc, prec=1)

# Plot accuracy over standard deviations (narrow range)
plt.figure(2)
plt.plot(sd_list, accuracy_history)
plt.axhline(y=50, color='r', linestyle='--')
plt.grid(True)
plt.xlim(min(sd_list), max(sd_list))
plt.ylim(0, max(accuracy_history)+10)
plt.title("Test Set Accuracy")
plt.xlabel("Noise (Standard Deviations)")
plt.ylabel("Classification Accuracy (%)")
