# Convolutional Neural Network for EEG classification

import os
import numpy as np
import network

from pylab import imshow, show, cm
from network import Network, shared, relu
from network import ConvLayer, ConvPoolLayer, FullyConnectedLayer, SoftmaxLayer

# Load EEG data
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
data_dir = os.path.join(parent_dir, "data")

data = np.load(os.path.join(data_dir, 'all_data_6_1d_full.npy'))

labels = np.load(os.path.join(data_dir, 'all_data_6_1d_full_labels.npy'))
labels = labels[:,1]

# Create train, validation, test sets
#rng = np.random.RandomState(225)
indices = np.random.permutation(data.shape[0])

split_train, split_val, split_test = .6, .2, .2

split_train = int(round(data.shape[0]*split_train))
split_val = split_train + int(round(data.shape[0]*split_val))

train_idx = indices[:split_train]
val_idx = indices[split_train:split_val]
test_idx = indices[split_val:]

tr_data = data[train_idx,:]
tr_labels = labels[train_idx]

val_data = data[val_idx,:]
val_labels = labels[val_idx]

te_data = data[test_idx,:]
te_labels = labels[test_idx]

train_data = shared((tr_data, tr_labels))
validation_data = shared((val_data, val_labels))
test_data = shared((te_data, te_labels))

# Show a single random trial
image_num = np.random.randint(0, network.size(train_data))
image_label = str(train_data[1][image_num].eval())
image_array = train_data[0][image_num].eval()
image_2d = np.reshape(image_array, (64, 512))

imshow(image_2d, cmap=cm.gray)
show()
print("Label: {}".format(image_label))

# Train
mini_batch_size = 10

def basic_conv(n=3, epochs=60):
    nets = []  # list of networks (for ensemble, if desired)
    for j in range(n):
        net = Network([
            ConvLayer(image_shape=(mini_batch_size, 1, 64, 512),
                      filter_shape=(20, 1, 3, 3), stride=(1, 1), activation_fn=relu),
            ConvPoolLayer(image_shape=(mini_batch_size, 20, 64, 512),
                          filter_shape=(40, 20, 3, 3), stride=(1, 1),
                          poolsize=(2, 2), activation_fn=relu),
            ConvPoolLayer(image_shape=(mini_batch_size, 40, 32, 256),
                          filter_shape=(80, 40, 3, 3), stride=(1, 1),
                          poolsize=(2, 2), activation_fn=relu),
            FullyConnectedLayer(n_in=80*16*128, n_out=100),
            SoftmaxLayer(n_in=100, n_out=2)],
            mini_batch_size, 50)
            
        net.SGD(train_data, epochs, mini_batch_size, 0.1,
                validation_data, test_data, lmbda=0.0)
                
        nets.append(net)  # Add current network to list
    return nets

conv_net = basic_conv(n=1, epochs=2)

# Plot training curve for 1 network
conv_net[0].plot_training_curve()

# Plot validation/test accuracy curve for 1 network
conv_net[0].plot_accuracy_curve()

# Create a plot of the learned filters for first conv layer
conv_net[0].layers[0].plot_filters(4, 5, "Filters - Layer 1")  # 20 filters
