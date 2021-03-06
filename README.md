# Python Machine Learning

Code samples for a directed studies course on computational modelling (UBC PSYC 547E)

This work was done primarily for learning purposes. The following have been implemented:

* Linear regression (using base Python)
* Logistic Regression (AND gate)
* Logistic Regression (OR gate)
* ANN (XOR gate)
* CNN (MNIST - hand written digits)
* CNN (CIFAR10 - colour objects)
* CNN (EEG - mind wandering classification)

**Note**: the code has only been tested on Windows 10 with Python 2.7.12 (64 bit)

## Requirements

* Python 2.7.x
 * Install: https://www.python.org/downloads/
    * This is should already be installed on Mac OSX
* [numpy](http://www.numpy.org/)
  * Install: `pip install numpy`
  * Used to handle arrays and matrices in Python
* [theano](http://deeplearning.net/software/theano/)
  * Install: `pip install theano`
  * Used for symbolic expressions and GPU training
* [matplotlib](http://matplotlib.org/)
  * Install: `pip install matplotlib`
  * Used for plotting
* [lasagne](http://lasagne.readthedocs.io/)
  * Install: `pip install --upgrade https://github.com/Lasagne/Lasagne/archive/master.zip`
  * This is only used for the Lasagne implementations of the different models
* [nolearn](https://pythonhosted.org/nolearn/)
  * Install: `pip install nolearn`
  * This is only used for the lasagne + nolearn implementations of the different models

The quickest/easiest way to get up and running is by installing the [Anaconda Python distribution](https://www.continuum.io/downloads), which comes with all dependencies installed (`theano`, `lasagne`, and `nolearn` will still need to be installed separately).
