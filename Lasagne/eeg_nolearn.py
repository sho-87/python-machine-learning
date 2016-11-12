from __future__ import print_function

import os
import theano
import theano.tensor as T
import numpy as np

from lasagne.layers import DenseLayer
from lasagne.layers import InputLayer
from lasagne.layers import DropoutLayer
from lasagne.layers import Conv2DLayer
from lasagne.layers import MaxPool2DLayer
from lasagne.nonlinearities import softmax
from lasagne.updates import adam
from lasagne.layers import get_all_params

from nolearn.lasagne import NeuralNet
from nolearn.lasagne import TrainSplit
from nolearn.lasagne import BatchIterator
from nolearn.lasagne.visualize import plot_loss
from nolearn.lasagne.visualize import plot_conv_weights
from nolearn.lasagne.visualize import plot_conv_activity
from nolearn.lasagne.visualize import plot_occlusion
from nolearn.lasagne.visualize import plot_saliency
from nolearn.lasagne import PrintLayerInfo

UPSAMPLE = True

# Load EEG data
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
data_dir = os.path.join(parent_dir, "data")

data = np.load(os.path.join(data_dir, 'all_data_1_2d_full.npy'))
data = data.reshape(-1, 1, 64, 512)

data_labels = np.load(os.path.join(data_dir, 'all_data_1_2d_full_labels.npy'))
data_labels = data_labels[:,1]

# Upsample the under-represented MW class
if UPSAMPLE:
    mw_idx = np.where(data_labels==1)
    mw_data = data[mw_idx]
    mw_data_labels = data_labels[mw_idx]
    
    num_mw = len(mw_idx[0])
    num_ot = data.shape[0] - num_mw
    
    num_to_bootstrap = num_ot - num_mw
    bootstrap_idx = np.random.randint(mw_data.shape[0], size=num_to_bootstrap)
    mw_data_boot = mw_data[bootstrap_idx]
    mw_data_labels_boot = mw_data_labels[bootstrap_idx]
    
    data = np.concatenate((data, mw_data_boot), axis=0)
    data_labels = np.concatenate((data_labels, mw_data_labels_boot), axis=0)

# Shuffle the data
indices = np.random.permutation(data.shape[0])
test_threshold = 0.1  # Proportion to save for test set
threshold_idx = int(round(data.shape[0]*(1-test_threshold)))
train_idx = indices[:threshold_idx]
test_idx = indices[threshold_idx:]

x_train = data[train_idx]
y_train = data_labels[train_idx]

x_test = data[test_idx]
y_test = data_labels[test_idx]

# Define architecture
layers0 = [
    # layer dealing with the input data
    (InputLayer, {'shape': (None, x_train.shape[1], x_train.shape[2], x_train.shape[3])}),

    # first stage of our convolutional layers
    (Conv2DLayer, {'num_filters': 20, 'filter_size': (1, 9)}),
    (MaxPool2DLayer, {'pool_size': (1, 4)}),

    # two dense layers with dropout
    (DenseLayer, {'num_units': 64}),

    # the output layer
    (DenseLayer, {'num_units': 2, 'nonlinearity': softmax}),
]

# Network parameters
net0 = NeuralNet(
    layers=layers0,
    max_epochs=30,
    batch_iterator_train = BatchIterator(batch_size=5, shuffle=True),

    update=adam,
    update_learning_rate=0.1,

    objective_l2=0.001,

    train_split=TrainSplit(eval_size=0.25),
    verbose=2,
)

# Train
net0.fit(x_train, y_train)

# Plot learning curve
plot_loss(net0)

# Plot learned filters
#plot_conv_weights(net0.layers_[1], figsize=(4, 4))  # Layer 1 (conv1)

# Plot activation maps
#x = x_train[0:1]
#plot_conv_activity(net0.layers_[1], x)

# Show filter occlusion maps to detect importance
#plot_occlusion(net0, x_train[:5], y_train[:5])

#plot_saliency(net0, x_train[:5])

#layer_info = PrintLayerInfo()
#layer_info(net0)

# Predict a label
print(net0.predict(x_test))
