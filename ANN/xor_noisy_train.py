# XOR gate. ANN with 1 hidden layer. Noise added to the train and test set

import numpy as np
import theano
import theano.tensor as T
import matplotlib.pyplot as plt

# Set inputs and correct output values
training_reps = 10
inputs = [[0,0], [1,1], [0,1], [1,0]] * training_reps
inputs = np.asarray(inputs, dtype='float32')  # Cast to float32
outputs = [0, 0, 1, 1] * training_reps

# Set training parameters
alpha = 0.1  # Learning rate
training_iterations = 50000
hidden_layer_nodes = 3  # Number of nodes in the hidden layer

# Set random seed
rng = np.random.RandomState(5353)

# Training loop - different model for each noise amount
training_noise = (0.2, 0.4, 0.6, 0.8, 1.0)  # SD of training noise distribution
test_noise = (0.2, 0.4, 0.6, 0.8)  # SD of test noise distribution
accuracy_history = {k: [] for k in test_noise}  # Intialize accuracy dict lists

for sd_train in training_noise:
    print "-- Training noise: {} SD --".format(sd_train)
    train_noise = rng.normal(0, sd_train, (len(inputs), 2)).astype('float32')
    train_data = np.add(inputs, train_noise)  # Add noise to original input

    # Define tensors
    x = T.matrix("x")
    y = T.vector("y")
    b1 = theano.shared(value=1.0, name='b1')
    b2 = theano.shared(value=1.0, name='b2')

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

    # Train
    cost_history = []

    for i in range(training_iterations):
        if (i+1) % 5000 == 0:
            print "Iteration #{} | Cost: {}".format(i+1, cost)
        h, cost = train(train_data, outputs)
        cost_history.append(cost)

    # Plot training curve
    plt.figure(0)
    plt.plot(range(1, len(cost_history)+1), cost_history, label=str(sd_train))
    plt.grid(True)
    plt.xlim(1, len(cost_history))
    plt.ylim(0, max(cost_history))
    plt.title("Training Curve")
    plt.xlabel("Iteration #")
    plt.ylabel("Cost")
    plt.legend(title="Noise (SD)", loc='best')

    # Predictions on noisy test data
    num_reps = 2000  # Number of times to repeat the test set
    labels = [0, 0, 1, 1] * num_reps  # Correct outputs
    raw_input = [[0,0], [1,1], [0,1], [1,0]] * num_reps
    raw_input = np.asarray(raw_input, dtype='float32')  # Cast to float32

    for sd_test in test_noise:
        # Add random noise to the input data (sampled from normal distribution)
        noise = rng.normal(0, sd_test, (len(raw_input), 2)).astype('float32')
        pred_data = np.add(raw_input, noise)

        predictions = predict(pred_data)[0]
        predictions[predictions >= 0.5] = 1
        predictions[predictions < 0.5] = 0

        # Calculate % accuracy for this SD set
        cur_acc = float(np.sum(predictions == labels))/len(predictions) * 100
        accuracy_history[sd_test].append(cur_acc)

# Plot test curve
plt.figure(1)
max_acc = 0

# Add plot for each test set SD value (sorted in order of key value)
for sd in sorted(accuracy_history.iterkeys()):
    acc = accuracy_history[sd]
    plt.plot(training_noise, acc, label=str(sd))
    max_acc = max(acc) if max(acc) > max_acc else max_acc

plt.grid(True)
plt.xlim(min(training_noise), max(training_noise))
plt.ylim(0, max_acc + 10)
plt.title("Test Set Accuracy")
plt.xlabel("Training Noise (Standard Deviation)")
plt.ylabel("Classification Accuracy (%)")
plt.legend(title='Test Noise (SD)', loc='best')
