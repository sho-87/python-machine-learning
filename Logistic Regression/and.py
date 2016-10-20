# AND gate. No hidden layer, basic logistic regression

import numpy as np
import theano
import theano.tensor as T
import matplotlib.pyplot as plt

# Set inputs and correct output values
inputs = [[0,0], [1,1], [0,1], [1,0]]
outputs = [0, 1, 0, 0]

# Set training parameters
alpha = 0.1  # Learning rate
training_iterations = 30000

 # Define tensors
x = T.matrix("x")
y = T.vector("y")
b = theano.shared(value=1.0, name='b')

# Initialize random weights
w_values = np.asarray(
    np.random.uniform(low=-1, high=1, size=(2, 1)),
    dtype=theano.config.floatX)  # Force type to 32bit float for GPU
w = theano.shared(value=w_values, name='w', borrow=True)

# Theano symbolic expressions
hypothesis = T.nnet.sigmoid(T.dot(x, w) + b)  # Sigmoid/logistic activation
hypothesis = T.flatten(hypothesis)  # This needs to be flattened so
                                    # hypothesis (matrix) and
                                    # y (vector) have the same shape

# cost = T.sum((y - hypothesis) ** 2)  # Quadratic/squared error loss
# cost = -(y*T.log(hypothesis) + (1-y)*T.log(1-hypothesis)).sum()  # Manual CE
# cost = T.nnet.categorical_crossentropy(hypothesis, y)  # Categorical CE
cost = T.nnet.binary_crossentropy(hypothesis, y).mean()  # Binary CE
updates_rules = [
    (w, w - alpha * T.grad(cost, wrt=w)),
    (b, b - alpha * T.grad(cost, wrt=b))
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
test_data = [[1,1], [1,1], [1,1], [1,0]]
predictions = predict(test_data)
print predictions
