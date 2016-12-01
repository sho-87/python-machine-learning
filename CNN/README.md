# Convolutional Neural Network

* **network.py**
  * Module containing classes and methods for different layer types
  * Modified version of the one created by Michael Nielson. The original work can be found [here](https://github.com/mnielsen/neural-networks-and-deep-learning)

* **mnist.py**
  * CNN for hand written digit recognition (MNIST) using `network.py`

* **mnist_lasagne.py**
  * CNN for hand written digit recognition (MNIST)
  * This is the same code as the Lasagne MNIST example with a few slight modifications

* **mnist_nolearn.py**
  * CNN for hand written digit recognition (MNIST)
  * Uses the nolearn package as a wrapper around lasagne and theano for learning diagnostics

* **cifar10.py**
  * CNN for colour object classification (CIFAR-10)

* **eeg.py**
  * CNN for classifying mind wandering from EEG data using `network.py`

* **eeg_nolearn.py**
  * CNN for EEG classification
  * Uses the nolearn package as a wrapper around lasagne and theano for learning diagnostics
