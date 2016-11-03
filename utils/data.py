import os
import numpy as np
import gzip
import time

from struct import unpack

def load_mnist(imagefile, labelfile):
    """Read input-vector (image) and target class (label, 0-9) and return
       it as list of tuples.
    """
    # Open the images with gzip in read binary mode
    images = gzip.open(imagefile, 'rb')
    labels = gzip.open(labelfile, 'rb')

    # Read the binary data

    # We have to get big endian unsigned int. So we need '>I'

    # Get metadata for images
    images.read(4)  # skip the magic_number
    number_of_images = images.read(4)
    number_of_images = unpack('>I', number_of_images)[0]
    rows = images.read(4)
    rows = unpack('>I', rows)[0]
    cols = images.read(4)
    cols = unpack('>I', cols)[0]

    # Get metadata for labels
    labels.read(4)  # skip the magic_number
    N = labels.read(4)
    N = unpack('>I', N)[0]

    if number_of_images != N:
        raise Exception('number of labels did not match the number of images')

    # Get the data
    x = np.zeros((N, rows, cols), dtype=np.uint8)  # Initialize numpy array
    y = np.zeros((N, 1), dtype=np.uint8)  # Initialize numpy array
    
    start_time = time.time()
    for i in range(N):
        if i % 1000 == 0:
            print "{filename} | {current}/{total} ({p:.2f}%) | Elapsed Time: {time:.2f}s".format(
                filename = os.path.basename(imagefile),
                current = i,
                total = number_of_images,
                p = (float(i)/number_of_images)*100,
                time = time.time() - start_time)

        for row in range(rows):
            for col in range(cols):
                tmp_pixel = images.read(1)  # Just a single byte
                tmp_pixel = unpack('>B', tmp_pixel)[0]
                x[i][row][col] = tmp_pixel
        tmp_label = labels.read(1)
        y[i] = unpack('>B', tmp_label)[0]
    return (x, y)
    
def save_mnist():
    """Save loaded MNIST data to disk"""
    
    parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    data_dir = os.path.join(parent_dir, 'data')

    print "Loading training data..."
    train_data, train_labels = load_mnist(
        os.path.join(data_dir, 'train-images-idx3-ubyte.gz'),
        os.path.join(data_dir, 'train-labels-idx1-ubyte.gz'))
        
    print "Loading test data..."
    test_data, test_labels = load_mnist(
        os.path.join(data_dir, 't10k-images-idx3-ubyte.gz'),
        os.path.join(data_dir, 't10k-labels-idx1-ubyte.gz'))
    
    # Save to disk
    np.save(os.path.join(data_dir, "mnist_train_data"), train_data)
    np.save(os.path.join(data_dir, "mnist_train_labels"), train_labels)
    np.save(os.path.join(data_dir, "mnist_test_data"), test_data)
    np.save(os.path.join(data_dir, "mnist_test_labels"), test_labels)
    
    print "Save complete"
    
def scale_standardize(features):
    """Standardize the input features.
    
    Parameters:
    features -- mxn matrix, where m is the number of training examples, and n is the number of features/variables. No bias column should be included
    
    Returns:
    x -- feature matrix with standardized values
    """
    
    x = (features - np.mean(features, axis=0)) / np.std(features, axis=0)
    x = np.insert(x, 0, 1, axis=1)  # Add 1's for bias

    return x
    
if __name__ == "__main__":
    save_mnist()