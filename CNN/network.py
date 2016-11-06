#### Libraries
# Standard library
import cPickle
import gzip
import time

# Third-party libraries
import numpy as np
import matplotlib.pyplot as plt
import theano
import theano.tensor as T
from theano.tensor.nnet import conv2d
from theano.tensor.nnet import softmax
from theano.tensor import shared_randomstreams
from theano.tensor.signal import pool
from pylab import cm

# Activation functions for neurons
def linear(z): return z
def ReLU(z): return T.maximum(0.0, z)
from theano.tensor.nnet import sigmoid
from theano.tensor import tanh

# Constants
GPU = True
if GPU:
    try: theano.config.device = 'gpu'
    except: pass # it's already set
    theano.config.floatX = 'float32'
else:
    print "Running with a CPU."

# Load the MNIST data
def load_mnist_shared(filename="../data/mnist.pkl.gz"):
    """Load MNIST data from file"""
    f = gzip.open(filename, 'rb')
    training_data, validation_data, test_data = cPickle.load(f)
    f.close()
    
    def shared(data):
        """Place the data into shared variables.
        
        This allows Theano to copy the data to the GPU, if one is available.
        """
        shared_x = theano.shared(
            np.asarray(data[0], dtype=theano.config.floatX), borrow=True)
        shared_y = theano.shared(
            np.asarray(data[1], dtype=theano.config.floatX), borrow=True)
        return shared_x, T.cast(shared_y, "int32")
    
    return [shared(training_data), shared(validation_data), shared(test_data)]

# Main class used to construct and train networks
class Network(object):

    def __init__(self, layers, mini_batch_size):
        """Takes a list of `layers`, describing the network architecture, and
        a value for the `mini_batch_size` to be used during training
        by stochastic gradient descent.

        """
        self.layers = layers
        self.mini_batch_size = mini_batch_size
        
        # Get all parameters across all layers
        self.params = [param for layer in self.layers for param in layer.params]
        
        self.x = T.matrix("x")
        self.y = T.ivector("y")
        
        # Setup the first layer of the network
        init_layer = self.layers[0]
        init_layer.set_inpt(self.x, self.x, self.mini_batch_size)
        
        # Set subsequent layers with inputs corresponding to
        # outputs from previous layer
        for j in xrange(1, len(self.layers)):
            prev_layer, layer  = self.layers[j-1], self.layers[j]
            layer.set_inpt(prev_layer.output,
                           prev_layer.output_dropout,
                           self.mini_batch_size)
                
        # Set output for the whole network (= final layer's output)
        self.output = self.layers[-1].output
        self.output_dropout = self.layers[-1].output_dropout
        
        # Initialize empty dicts for storing history values
        self.cost_history = {"iteration": [], "cost": []}
        self.accuracy_history = {"validation": {"epoch": [], "score":[]},
                            "test": {"epoch": [], "score":[]}}
                            
    def plot_training_curve(self):
        """ Plot training curve """
        plt.plot(self.cost_history["iteration"], self.cost_history["cost"])
        plt.grid(True)
        plt.title("Training Curve")
        plt.xlabel("Iteration #")
        plt.ylabel("Cost")
        plt.show()
        
    def plot_accuracy_curve(self):
        """ Plot accuracy curve """
        plt.plot(self.accuracy_history["validation"]["epoch"],
                 self.accuracy_history["validation"]["score"],
                label="Validation")
        plt.plot(self.accuracy_history["test"]["epoch"],
                 self.accuracy_history["test"]["score"],
                label="Test")
        plt.grid(True)
        plt.title("Accuracy Curve")
        plt.xlabel("Epoch #")
        plt.ylabel("Accuracy")
        plt.legend(loc='best')
        plt.show()

    def SGD(self, training_data, epochs, mini_batch_size, alpha,
            validation_data, test_data, lmbda=0.0):
        """Train the network using mini-batch stochastic gradient descent."""
        training_x, training_y = training_data
        validation_x, validation_y = validation_data
        test_x, test_y = test_data

        # compute number of minibatches for training, validation and testing
        num_training_batches = size(training_data)/mini_batch_size
        num_validation_batches = size(validation_data)/mini_batch_size
        num_test_batches = size(test_data)/mini_batch_size

        # define (regularized) cost function, symbolic gradients, and updates
        l2_norm_squared = sum([(layer.w**2).sum() for layer in self.layers])
        cost = self.layers[-1].cost(self)+\
               0.5*lmbda*l2_norm_squared/num_training_batches
        grads = T.grad(cost, self.params)
        updates = [(param, param - alpha*grad)
                   for param, grad in zip(self.params, grads)]

        # define functions to train a mini-batch, and to compute the
        # accuracy in validation and test mini-batches.
        i = T.lscalar() # mini-batch index
        
        train_mb = theano.function(
            [i], cost, updates=updates,
            givens={
                self.x:
                training_x[i*self.mini_batch_size: (i+1)*self.mini_batch_size],
                self.y:
                training_y[i*self.mini_batch_size: (i+1)*self.mini_batch_size]
            })
            
        validate_mb_accuracy = theano.function(
            [i], self.layers[-1].accuracy(self.y),
            givens={
                self.x:
                validation_x[i*self.mini_batch_size: (i+1)*self.mini_batch_size],
                self.y:
                validation_y[i*self.mini_batch_size: (i+1)*self.mini_batch_size]
            })
            
        test_mb_accuracy = theano.function(
            [i], self.layers[-1].accuracy(self.y),
            givens={
                self.x:
                test_x[i*self.mini_batch_size: (i+1)*self.mini_batch_size],
                self.y:
                test_y[i*self.mini_batch_size: (i+1)*self.mini_batch_size]
            })
            
        self.test_mb_predictions = theano.function(
            [i], self.layers[-1].y_out,
            givens={
                self.x:
                test_x[i*self.mini_batch_size: (i+1)*self.mini_batch_size]
            })
            
        # Do the actual training
        best_validation_accuracy = 0.0                            
        start_time = time.time()
                            
        for epoch in xrange(epochs):
            for minibatch_index in xrange(num_training_batches):
                iteration = num_training_batches * epoch + minibatch_index

                cost_ij = train_mb(minibatch_index)
                
                if iteration % 1000 == 0:
                    print("Training mini-batch {0}/{1} | Cost: {2:.4f} | Elapsed time: {3:.2f}s".format(
                        iteration,
                        num_training_batches * epochs,
                        float(cost_ij),
                        time.time() - start_time))
                        
                    # Store history
                    self.cost_history["iteration"].append(iteration)
                    self.cost_history["cost"].append(cost_ij)

                if (iteration+1) % num_training_batches == 0:
                    validation_accuracy = np.mean(
                        [validate_mb_accuracy(j) for j in xrange(num_validation_batches)])
                    
                    # Store history
                    self.accuracy_history["validation"]["epoch"].append(epoch)                 
                    self.accuracy_history["validation"]["score"].append(validation_accuracy)     
                        
                    print("Epoch {0}: validation accuracy {1:.2%}".format(
                        epoch, validation_accuracy))
                        
                    if validation_accuracy >= best_validation_accuracy:
                        print("This is the best validation accuracy to date.")
                        best_validation_accuracy = validation_accuracy
                        best_iteration = iteration
                        
                    if test_data:
                        test_accuracy = np.mean(
                            [test_mb_accuracy(j) for j in xrange(num_test_batches)])
                        
                        # Store history
                        self.accuracy_history["test"]["epoch"].append(epoch)                 
                        self.accuracy_history["test"]["score"].append(test_accuracy)
                
                        print('The corresponding test accuracy is {0:.2%}'.format(
                            test_accuracy))
        
        print("Finished training network.")
        print("Best validation accuracy of {0:.2%} obtained at iteration {1}".format(
            best_validation_accuracy, best_iteration))
        print("Corresponding test accuracy of {0:.2%}".format(test_accuracy))

# Define layer types

class ConvPoolLayer(object):
    """Used to create a combination of a convolutional and a max-pooling
    layer.  A more sophisticated implementation would separate the
    two, but for our purposes we'll always use them together, and it
    simplifies the code, so it makes sense to combine them.

    """

    def __init__(self, filter_shape, image_shape, border_mode='half',
                 stride=(1, 1), poolsize=(2, 2), activation_fn=sigmoid):
        """`filter_shape` is a tuple of length 4, whose entries are the number
        of filters, the number of input feature maps, the filter height, and
        the filter width.

        `image_shape` is a tuple of length 4, whose entries are the
        mini-batch size, the number of input feature maps, the image
        height, and the image width.

        `poolsize` is a tuple of length 2, whose entries are the y and
        x pooling sizes.

        """
        self.filter_shape = filter_shape
        self.image_shape = image_shape
        self.border_mode = border_mode
        self.stride = stride
        self.poolsize = poolsize
        self.activation_fn = activation_fn
        
        # initialize weights and biases
        n_in = np.prod(filter_shape[1:])  # Total number of input params
        # n_out = (filter_shape[0]*np.prod(filter_shape[2:])/np.prod(poolsize))
        self.w = theano.shared(
            np.asarray(
                np.random.normal(loc=0, scale=np.sqrt(1.0/n_in), size=filter_shape),
                dtype=theano.config.floatX),
            borrow=True)
        self.b = theano.shared(
            np.asarray(
                np.random.normal(loc=0, scale=1.0, size=(filter_shape[0],)),
                dtype=theano.config.floatX),
            borrow=True)
        self.params = [self.w, self.b]

    def set_inpt(self, inpt, inpt_dropout, mini_batch_size):
        # Reshape the input to 2D
        self.inpt = inpt.reshape(self.image_shape)
        
        # Do convolution
        self.conv_out = conv2d(
            input=self.inpt, filters=self.w, filter_shape=self.filter_shape,
            input_shape=self.image_shape, border_mode=self.border_mode,
            subsample=self.stride)
        
        # Get the feature maps for this layer
        self.feature_maps = theano.function([self.inpt], self.conv_out)
        
        # Max pooling
        pooled_out = pool.pool_2d(input=self.conv_out, ds=self.poolsize,
                                  ignore_border=True, mode='max')
            
        # Apply bias and activation and set as output
        self.output = self.activation_fn(
            pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))
        self.output_dropout = self.output # no dropout in convolutional layers
        
    def plot_filters(self, x, y, title, cmap=cm.gray):
        """Plot the filters after the (convolutional) layer.
        
        They are plotted in x by y format.  So, for example, if we
        have 20 filters in the layer, then we can call 
        plot_filters(4, 5, "title") to get a 4 by 5 plot of all layer filters.
        """
        filters = self.w.eval()  # Get filter values/weights
        
        fig = plt.figure()
        fig.suptitle(title)
        
        for j in range(len(filters)):
            ax = fig.add_subplot(x, y, j+1)
            ax.matshow(filters[j][0], cmap=cmap)
            plt.xticks(np.array([]))
            plt.yticks(np.array([]))
        
        plt.tight_layout()
        fig.subplots_adjust(top=0.90)
        plt.show()

class FullyConnectedLayer(object):

    def __init__(self, n_in, n_out, activation_fn=sigmoid, p_dropout=0.0):
        self.n_in = n_in
        self.n_out = n_out
        self.activation_fn = activation_fn
        self.p_dropout = p_dropout
        # Initialize weights and biases
        self.w = theano.shared(
            np.asarray(
                np.random.normal(
                    loc=0.0, scale=np.sqrt(1.0/n_in), size=(n_in, n_out)),
                dtype=theano.config.floatX),
            name='w', borrow=True)
        self.b = theano.shared(
            np.asarray(np.random.normal(loc=0.0, scale=1.0, size=(n_out,)),
                       dtype=theano.config.floatX),
            name='b', borrow=True)
        self.params = [self.w, self.b]

    def set_inpt(self, inpt, inpt_dropout, mini_batch_size):
        self.inpt = inpt.reshape((mini_batch_size, self.n_in))
        self.output = self.activation_fn(
            (1-self.p_dropout)*T.dot(self.inpt, self.w) + self.b)
        self.y_out = T.argmax(self.output, axis=1)
        self.inpt_dropout = dropout_layer(
            inpt_dropout.reshape((mini_batch_size, self.n_in)), self.p_dropout)
        self.output_dropout = self.activation_fn(
            T.dot(self.inpt_dropout, self.w) + self.b)

    def accuracy(self, y):
        "Return the accuracy for the mini-batch."
        return T.mean(T.eq(y, self.y_out))

class SoftmaxLayer(object):

    def __init__(self, n_in, n_out, p_dropout=0.0):
        self.n_in = n_in
        self.n_out = n_out
        self.p_dropout = p_dropout
        # Initialize weights and biases
        self.w = theano.shared(
            np.zeros((n_in, n_out), dtype=theano.config.floatX),
            name='w', borrow=True)
        self.b = theano.shared(
            np.zeros((n_out,), dtype=theano.config.floatX),
            name='b', borrow=True)
        self.params = [self.w, self.b]

    def set_inpt(self, inpt, inpt_dropout, mini_batch_size):
        self.inpt = inpt.reshape((mini_batch_size, self.n_in))
        self.output = softmax((1-self.p_dropout)*T.dot(self.inpt, self.w) + self.b)
        self.y_out = T.argmax(self.output, axis=1)  # Predicted class
        self.inpt_dropout = dropout_layer(
            inpt_dropout.reshape((mini_batch_size, self.n_in)), self.p_dropout)
        self.output_dropout = softmax(T.dot(self.inpt_dropout, self.w) + self.b)

    def cost(self, net):
        "Return the log-likelihood cost."
        return -T.mean(T.log(self.output_dropout)[T.arange(net.y.shape[0]), net.y])

    def accuracy(self, y):
        "Return the accuracy for the mini-batch."
        return T.mean(T.eq(y, self.y_out))


# Miscellanea
def size(data):
    "Return the size of the dataset `data`."
    return data[0].get_value(borrow=True).shape[0]

def dropout_layer(layer, p_dropout):
    srng = shared_randomstreams.RandomStreams(
        np.random.RandomState(0).randint(999999))
    mask = srng.binomial(n=1, p=1-p_dropout, size=layer.shape)
    return layer*T.cast(mask, theano.config.floatX)
