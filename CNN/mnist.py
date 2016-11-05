# Convolutional Neural Network for handwritten digit classification (MNIST)

import numpy as np
import network

from pylab import imshow, show, cm
from network import sigmoid, tanh, ReLU, Network
from network import ConvPoolLayer, FullyConnectedLayer, SoftmaxLayer

# Load MNIST data
training_data, validation_data, test_data = network.load_data_shared()

## Debug: manually inspect the loaded mnist data
#import gzip
#import cPickle
#f = gzip.open('../data/mnist.pkl.gz', 'rb')
#training_data, validation_data, test_data = cPickle.load(f)
#f.close()

# Show a single random image
image_num = np.random.randint(0, network.size(training_data))
image_label = str(training_data[1][image_num].eval())
image_array = training_data[0][image_num].eval()
image_2d = np.reshape(image_array, (28, 28))

imshow(image_2d, cmap=cm.gray)
show()
print("Label: {}".format(image_label))

# Train
mini_batch_size = 10

def basic_conv(n=3, epochs=60):
    nets = []  # list of networks (for ensemble, if desired)
    for j in range(n):
        net = Network([
            ConvPoolLayer(image_shape=(mini_batch_size, 1, 28, 28), 
                          filter_shape=(20, 1, 5, 5), 
                          poolsize=(2, 2)),
            FullyConnectedLayer(n_in=20*12*12, n_out=100),
            SoftmaxLayer(n_in=100, n_out=10)], mini_batch_size)
            
        net.SGD(training_data, epochs, mini_batch_size, 0.1, validation_data, test_data)
        nets.append(net)
    return nets

conv_net = basic_conv(n=1, epochs=1)

# Plot training curve for 1 network
conv_net[0].plot_training_curve()

# Plot validation/test accuracy curve for 1 network
conv_net[0].plot_accuracy_curve()

# Create a plot of the learned filters for a conv layer
conv_net[0].layers[0].plot_filters(5, 4)
