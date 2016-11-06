import os
import theano
import network
import numpy as np
import theano.tensor as T

from pylab import imshow, show, cm
from network import sigmoid, tanh, ReLU, Network
from network import ConvPoolLayer, FullyConnectedLayer, SoftmaxLayer

# Load CIFAR-10 data
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
data_dir = os.path.join(parent_dir, "data")
data_dir_cifar10 = os.path.join(data_dir, "cifar-10-batches-py")
class_names_cifar10 = np.load(os.path.join(data_dir_cifar10, "batches.meta"))

def one_hot(x, n):
    """
    convert index representation to one-hot representation
    """
    x = np.array(x)
    assert x.ndim == 1
    return np.eye(n)[x]
    
def _grayscale(a):
    return a.reshape(a.shape[0], 3, 32, 32).mean(1).reshape(a.shape[0], -1)

def _load_batch_cifar10(filename, dtype='float32'):
    """
    load a batch in the CIFAR-10 format
    """
    path = os.path.join(data_dir_cifar10, filename)
    batch = np.load(path)
    data = batch['data'] / 255.0 # scale between [0, 1]
    labels = np.array(batch['labels']) # convert labels to one-hot representation
    return data.astype(dtype), labels.astype(dtype)
    
def cifar10(dtype='float32'):
    # train
    x_train = []
    t_train = []
    for k in xrange(5):
        x, t = _load_batch_cifar10("data_batch_%d" % (k + 1), dtype=dtype)
        x_train.append(x)
        t_train.append(t)

    x_train = np.concatenate(x_train, axis=0)
    t_train = np.concatenate(t_train, axis=0)

    # test
    x_test, t_test = _load_batch_cifar10("test_batch", dtype=dtype)

    return x_train, t_train, x_test, t_test

# Create validation set from test data
tr_data, tr_labels, te_data, te_labels = cifar10(dtype=theano.config.floatX)
val_data = te_data[:5000]  # FIXME: randomize validation set selection
val_labels = te_labels[:5000]
te_data = te_data[5000:]
te_labels = te_labels[5000:]

train_data = (tr_data, tr_labels)
validation_data = (val_data, val_labels)
test_data = (te_data, te_labels)

#labels_test = np.argmax(te_labels, axis=1)

# Load data onto GPU
theano.config.floatX = 'float32'

def shared(data):
    """Place the data into shared variables.
    
    This allows Theano to copy the data to the GPU, if one is available.
    """
    shared_x = theano.shared(
        np.asarray(data[0], dtype=theano.config.floatX), borrow=True)
    shared_y = theano.shared(
        np.asarray(data[1], dtype=theano.config.floatX), borrow=True)
    return shared_x, T.cast(shared_y, "int32")

train_data = shared(train_data)
validation_data = shared(validation_data)
test_data = shared(test_data)

# Show a single random image
image_num = np.random.randint(0, network.size(train_data))
image_label = train_data[1][image_num].eval()
image_label = class_names_cifar10["label_names"][image_label]
image_array = train_data[0][image_num].eval()
image_3d = image_array.reshape(3,32,32).transpose(1,2,0)

imshow(image_3d)
show()
print("Label: {}".format(image_label))

# Train
mini_batch_size = 10

def basic_conv(n=3, epochs=60):
    nets = []  # list of networks (for ensemble, if desired)
    for j in range(n):
        net = Network([
            ConvPoolLayer(image_shape=(mini_batch_size, 3, 32, 32), 
                          filter_shape=(32, 3, 3, 3), stride=(1, 1),
                          poolsize=(2, 2), activation_fn=ReLU),
            ConvPoolLayer(image_shape=(mini_batch_size, 32, 16, 16), 
                          filter_shape=(80, 32, 3, 3), stride=(1, 1),
                          poolsize=(2, 2), activation_fn=ReLU),
            ConvPoolLayer(image_shape=(mini_batch_size, 80, 8, 8), 
                          filter_shape=(128, 80, 3, 3), stride=(1, 1),
                          poolsize=(2, 2), activation_fn=ReLU),
            FullyConnectedLayer(n_in=128*4*4, n_out=100),
            SoftmaxLayer(n_in=100, n_out=10)], mini_batch_size)
            
        net.SGD(train_data, epochs, mini_batch_size, 0.1,
                validation_data, test_data)
                
        nets.append(net)  # Add current network to list
    return nets

conv_net = basic_conv(n=1, epochs=2)

# Plot training curve for 1 network
conv_net[0].plot_training_curve()

# Plot validation/test accuracy curve for 1 network
conv_net[0].plot_accuracy_curve()

# Create a plot of the learned filters for first conv layer
conv_net[0].layers[0].plot_filters(6, 6, "Filters - Layer 1")
