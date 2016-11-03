# Convolutional Neural Network for handwritten digit classification (MNIST)

import os
import numpy as np
import theano

from pylab import imshow, show, cm

def display_mnist_digit(image, label=""):
    """View a single image."""
    print "Label: {}".format(label)
    imshow(image, cmap=cm.gray)
    show()

# Paths
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
data_dir = os.path.join(parent_dir, 'data')
utils_dir = os.path.join(parent_dir, 'utils')

# Load MNIST data
train_data = np.load(os.path.join(data_dir, "mnist_train_data.npy"))
train_labels = np.load(os.path.join(data_dir, "mnist_train_labels.npy"))
test_data = np.load(os.path.join(data_dir, "mnist_test_data.npy"))
test_labels = np.load(os.path.join(data_dir, "mnist_test_labels.npy"))
    
# Show a single random image
image_num = np.random.randint(0, len(train_data))
display_mnist_digit(train_data[image_num,:,:], train_labels[image_num][0])
