# Python Machine Learning

Code samples for a directed studies course on computational modelling (UBC PSYC 547E)

This work was done primarily for learning purposes. The following have been implemented:

* Linear regression
* Logistic regression
* Neural network (for learning an AND gate)
* Neural network (for learning an OR gate)
* Neural network (for learning an XOR gate)

**Note**: the code has only been tested on Windows 10 with Python 2.7.12 (64 bit)

## Requirements

* Python 2.7.x
 * Install: https://www.python.org/downloads/
    * This is should already be installed on Mac OSX
* [numpy](http://www.numpy.org/)
  * Install: `pip install numpy`
  * Used to handle arrays and matrices in Python
* [matplotlib](http://matplotlib.org/)
  * Install: `pip install matplotlib`
  * Used for plotting

The quickest/easiest way to get up and running is by installing the [Anaconda Python distribution](https://www.continuum.io/downloads), which comes with all dependencies installed.

## To Do

* Refactor code into classes for modularity
* (Better) Integration of [Theano](http://deeplearning.net/software/theano/) so we can...
* Outsource work to the GPU (if available)
