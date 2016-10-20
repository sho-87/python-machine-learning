# XOR gate. ANN with 1 hidden layer

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
        print "Iteration #%s: " % str(i+1)
        print "Cost: %s" % str(cost)
    h, cost = train(inputs, outputs)
    cost_history.append(cost)

# Plot training curve
plt.plot(range(1, len(cost_history)+1), cost_history)
plt.grid(True)
plt.xlim(1, len(cost_history))
plt.ylim(0, max(cost_history))
plt.title("Training Curve")
plt.xlabel("Iteration #")
plt.ylabel("Cost")

# Predictions
test_data = [[0,0], [1,1], [1,1], [1,0]]
predictions = predict(test_data)
print predictions