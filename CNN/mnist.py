# Convolutional Neural Network for handwritten digit classification (MNIST)

import numpy as np
import network
import matplotlib.pyplot as plt

from pylab import imshow, show, cm
from network import sigmoid, tanh, ReLU, Network
from network import ConvPoolLayer, FullyConnectedLayer, SoftmaxLayer

# Load MNIST data
training_data, validation_data, test_data = network.load_mnist_shared()

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
                          filter_shape=(20, 1, 5, 5), stride=(1, 1),
                          poolsize=(2, 2), activation_fn=ReLU),
            ConvPoolLayer(image_shape=(mini_batch_size, 20, 14, 14), 
                          filter_shape=(40, 20, 5, 5), stride=(1, 1),
                          poolsize=(2, 2), activation_fn=ReLU),
            FullyConnectedLayer(n_in=40*7*7, n_out=100),
            SoftmaxLayer(n_in=100, n_out=10)], mini_batch_size)
            
        net.SGD(training_data, epochs, mini_batch_size, 0.1,
                validation_data, test_data)
                
        nets.append(net)  # Add current network to list
    return nets

conv_net = basic_conv(n=1, epochs=1)

# Plot training curve for 1 network
conv_net[0].plot_training_curve()

# Plot validation/test accuracy curve for 1 network
conv_net[0].plot_accuracy_curve()

# Create a plot of the learned filters for first conv layer
conv_net[0].layers[0].plot_filters(4, 5, "Filters - Layer 1")  # 20 filters

# Plot feature maps after 1st convolution layer
image_4d = image_2d.reshape(1, 1, 28, 28)  # Reshape orig random image to 4D
activations = conv_net[0].layers[0].feature_maps(image_4d)
num_maps = activations.shape[1]  # Number of learned filters

feature_map_plot = plt.figure()
feature_map_plot.suptitle("Feature maps - Layer 1")

for j in range(num_maps):
    ax = feature_map_plot.add_subplot(4, 5, j+1)
    ax.matshow(activations[0][j], cmap = cm.gray)
    plt.xticks(np.array([]))
    plt.yticks(np.array([]))

plt.tight_layout()
feature_map_plot.subplots_adjust(top=0.90)
plt.show()

# Create a plot of the learned filters for second conv layer
conv_net[0].layers[1].plot_filters(5, 8, "Filters - Layer 2")  # 40 filters

# Feature maps for second convolutional layer are not plotted as they depend
# on activations from prior layers
