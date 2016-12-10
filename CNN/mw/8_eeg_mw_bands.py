from __future__ import print_function

import os
import time
import numpy as np
import theano
import theano.tensor as T
import lasagne
import matplotlib.pyplot as plt

from tqdm import tqdm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from lasagne.layers import InputLayer, Conv2DLayer, Pool2DLayer
from lasagne.regularization import regularize_network_params, l2

VERBOSE = False
GRID_SEARCH = False
NUM_ELECTRODES = 30  # 4 or 30 electrodes

def bootstrap(data, labels, boot_type="downsample"):
    print("Bootstrapping data...")
    ot_class = 0
    mw_class = 1
    
    ot_idx = np.where(labels == ot_class)
    mw_idx = np.where(labels == mw_class)
    
    # Get OT examples
    ot_data = data[ot_idx]
    ot_labels = labels[ot_idx]
    print(" - OT (class: {}) | Data: {} | Labels: {}".format(ot_class, ot_data.shape, ot_labels.shape))
    
    # Get MW examples
    mw_data = data[mw_idx]
    mw_labels = labels[mw_idx]
    print(" - MW (class: {}) | Data: {} | Labels: {}".format(mw_class, mw_data.shape, mw_labels.shape))
    
    # Set majority and minority classes
    if ot_data.shape[0] > mw_data.shape[0]:
        maj_class, maj_data, maj_labels = ot_class, ot_data, ot_labels
        min_class, min_data, min_labels = mw_class, mw_data, mw_labels
    else:
        maj_class, maj_data, maj_labels = mw_class, mw_data, mw_labels
        min_class, min_data, min_labels = ot_class, ot_data, ot_labels
    
    print(" - Majority class: {} (N = {}) | Minority class: {} (N = {})".format(maj_class, maj_data.shape[0],
          min_class, min_data.shape[0]))
    
    # Upsample minority class
    if boot_type == "upsample":
        print("Upsampling minority class...")
        
        num_to_boot = maj_data.shape[0] - min_data.shape[0]
        print(" - Number to upsample: {}".format(num_to_boot))
        
        bootstrap_idx = np.random.randint(min_data.shape[0], size=num_to_boot)

        min_data_boot = min_data[bootstrap_idx]
        min_labels_boot = min_labels[bootstrap_idx]
        
        final_data = np.concatenate((data, min_data_boot), axis=0)
        final_labels = np.concatenate((labels, min_labels_boot), axis=0)
    elif boot_type == "downsample":
        print("Downsampling majority class...")
        # Resample N = number of minority examples
        num_to_boot = min_data.shape[0]
        
        bootstrap_idx = np.random.randint(maj_data.shape[0], size=num_to_boot)
        
        maj_data_boot = maj_data[bootstrap_idx]
        maj_labels_boot = maj_labels[bootstrap_idx]
        
        final_data = np.concatenate((maj_data_boot, min_data), axis=0)
        final_labels = np.concatenate((maj_labels_boot, min_labels), axis=0)
        
    print("Final class balance: {} ({}) - {} ({})".format(
        maj_class, len(np.where(final_labels==maj_class)[0]),
        min_class, len(np.where(final_labels==min_class)[0])))
        
    return final_data, final_labels

# Load EEG data
base_dir = os.path.abspath(os.path.join(os.path.join(os.path.dirname(__file__), os.pardir), os.pardir))
data_dir = os.path.join(base_dir, "data")

data = np.load(os.path.join(data_dir, 'all_data_6_2d_full_30ch_bands.npy'))

data_labels = np.load(os.path.join(data_dir, 'all_data_6_2d_full_30ch_bands_labels.npy'))
data_labels = data_labels[:,1]

## EEG frequency bands were created directly from MATLAB data using:
#from scipy import signal
#
#FREQ_BANDS = ((0.1, 3), (4, 7), (8, 13), (16, 31)) # Delta, theta, alpha, beta
#SAMPLE_RATE = 256
#
#all_data = np.reshape(all_data, (-1, 1, 30, 512))
#
#filtered_data = [all_data]
#
#for band in FREQ_BANDS:
#    lower = float(band[0])/(SAMPLE_RATE/2)  # Proportion of Nyquist frequency
#    upper = float(band[1])/(SAMPLE_RATE/2)  # Proportion of Nyquist frequency
#    b, a = signal.butter(1, [lower, upper], 'bandpass')
#    
#    output_signal = signal.filtfilt(b, a, all_data, axis=3)
#    
#    filtered_data.append(output_signal)
#    
#all_data = np.concatenate(filtered_data, axis=1)


# Electrode Order (30 channels)
electrode_order_30 = ('Fp1','Fp2','Fz',
                 'F4','F8','FC6',
                 'C4','T8','CP6',
                 'P4','P8','P10',
                 'O2','Oz','O1',
                 'P9','P3','P7',
                 'CP5','C3','T7',
                 'FC5','F7','F3',
                 'FC1','FC2','Cz',
                 'CP1','CP2','Pz')

# Downsample to specific electrodes (4 channels)
electrode_order_4 = ('Fp1','Oz','T8','T7')

if NUM_ELECTRODES == 4:
    data_temp = np.empty((data.shape[0], data.shape[1], 4, data.shape[3])).astype('float32')
    
    for i in range(data.shape[0]):
        data_temp[i,:,0,:] = data[i,:,0,:]  # Fp1
        data_temp[i,:,1,:] = data[i,:,13,:]  # Oz
        data_temp[i,:,2,:] = data[i,:,7,:]  # T8
        data_temp[i,:,3,:] = data[i,:,20,:]  # T7
        
    data = data_temp
    del data_temp

# Plot frequency bands
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
ax1.title.set_text('Unstandardized')
ax1.grid(True)
ax1.plot(data[0,0,0,:], label="Raw")
ax1.plot(data[0,1,0,:], label="Delta")
ax1.plot(data[0,2,0,:], label="Theta")
ax1.plot(data[0,3,0,:], label="Alpha")
ax1.plot(data[0,4,0,:], label="Beta")
ax1.set_xlim([0, 512])
ax1.legend(loc='upper right')

# Standardize data per trial
# Significantly improves gradient descent
data = (data - data.mean(axis=(2,3),keepdims=1)) / data.std(axis=(2,3),keepdims=1)

ax2.title.set_text('Standardized')
ax2.grid(True)
ax2.plot(data[0,0,0,:], label="Raw")
ax2.plot(data[0,1,0,:], label="Delta")
ax2.plot(data[0,2,0,:], label="Theta")
ax2.plot(data[0,3,0,:], label="Alpha")
ax2.plot(data[0,4,0,:], label="Beta")
ax2.set_xlim([0, 512])
ax2.legend(loc='upper right')

plt.show()

# Up/downsample the data to balance classes
data, data_labels = bootstrap(data, data_labels, "downsample")

# Create train, validation, test sets
rng = np.random.RandomState(5334)  # Set random seed
indices = rng.permutation(data.shape[0])

split_train, split_val, split_test = .6, .2, .2

split_train = int(round(data.shape[0]*split_train))
split_val = split_train + int(round(data.shape[0]*split_val))

train_idx = indices[:split_train]
val_idx = indices[split_train:split_val]
test_idx = indices[split_val:]

train_data = data[train_idx,:]
train_labels = data_labels[train_idx]

val_data = data[val_idx,:]
val_labels = data_labels[val_idx]

test_data = data[test_idx,:]
test_labels = data_labels[test_idx]

def build_cnn(k_height=1, k_width=25, input_var=None):
    # Input layer, as usual:
    l_in = InputLayer(shape=(None, 5, NUM_ELECTRODES, 512), input_var=input_var)

    l_conv1 = Conv2DLayer(incoming = l_in, num_filters = 8,
                          filter_size = (k_height, k_width),
                          stride = 1, pad = 'same',
                          W = lasagne.init.Normal(std = 0.02),
                          nonlinearity = lasagne.nonlinearities.very_leaky_rectify)
                        
    l_pool1 = Pool2DLayer(incoming = l_conv1, pool_size = (2,2), stride = (2,2))

    l_drop1 = lasagne.layers.dropout(l_pool1, p=.75)
    
    l_fc = lasagne.layers.DenseLayer(
            l_drop1,
            num_units=50,
            nonlinearity=lasagne.nonlinearities.rectify)
            
    l_drop2 = lasagne.layers.dropout(l_fc, p=.75)

    l_out = lasagne.layers.DenseLayer(
            l_drop2,
            num_units=2,
            nonlinearity=lasagne.nonlinearities.softmax)

    return l_out
    
# ############################# Batch iterator ###############################
# This is just a simple helper function iterating over training data in
# mini-batches of a particular size, optionally in random order. It assumes
# data is available as numpy arrays. For big datasets, you could load numpy
# arrays as memory-mapped files (np.load(..., mmap_mode='r')), or write your
# own custom data iteration function. For small datasets, you can also copy
# them to GPU at once for slightly improved performance. This would involve
# several changes in the main program, though, and is not demonstrated here.
# Notice that this function returns only mini-batches of size `batchsize`.
# If the size of the data is not a multiple of `batchsize`, it will not
# return the last (remaining) mini-batch.

def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    # tqdm() can be removed if no visual progress bar is needed
    for start_idx in tqdm(range(0, len(inputs) - batchsize + 1, batchsize)):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]


def main(model='cnn', batch_size=500, num_epochs=500, k_height=1, k_width=25):
    # Prepare Theano variables for inputs and targets
    input_var = T.tensor4('inputs')
    target_var = T.ivector('targets')

    network = build_cnn(k_height, k_width, input_var)

    # Create a loss expression for training, i.e., a scalar objective we want
    # to minimize (for our multi-class problem, it is the cross-entropy loss):
    prediction = lasagne.layers.get_output(network)
    loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
    loss = loss.mean()
    # We could add some weight decay as well here, see lasagne.regularization.
    l2_reg = regularize_network_params(network, l2)
    loss += l2_reg * 0.001
    
    train_acc = T.mean(T.eq(T.argmax(prediction, axis=1), target_var),
                      dtype=theano.config.floatX)

    # Create update expressions for training
    params = lasagne.layers.get_all_params(network, trainable=True)    
    updates = lasagne.updates.nesterov_momentum(loss, params, learning_rate=0.01)
    #updates = lasagne.updates.adam(loss, params, learning_rate=0.1)

    # Create a loss expression for validation/testing. The crucial difference
    # here is that we do a deterministic forward pass through the network,
    # disabling dropout layers.
    test_prediction = lasagne.layers.get_output(network, deterministic=True)
    test_loss = lasagne.objectives.categorical_crossentropy(test_prediction,
                                                            target_var)
    test_loss = test_loss.mean()
    
    # As a bonus, also create an expression for the classification accuracy:
    test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var),
                      dtype=theano.config.floatX)

    # Compile a function performing a training step on a mini-batch (by giving
    # the updates dictionary) and returning the corresponding training loss:
    train_fn = theano.function([input_var, target_var], [loss, train_acc], updates=updates)

    # Compile a second function computing the validation loss and accuracy:
    val_fn = theano.function([input_var, target_var], [test_loss, test_acc])

    training_hist = []
    val_hist = []    
    
    print("Starting training...")
    # We iterate over epochs:
    for epoch in range(num_epochs):
        # In each epoch, we do a full pass over the training data:
        print("Training epoch {}...".format(epoch+1))
        train_err = 0
        train_acc = 0
        train_batches = 0
        start_time = time.time()
        for batch in iterate_minibatches(train_data, train_labels, batch_size, shuffle=True):
            inputs, targets = batch
            err, acc = train_fn(inputs, targets)
            train_err += err
            train_acc += acc
            train_batches += 1
            if VERBOSE:
                print("Epoch: {} | Mini-batch: {}/{} | Elapsed time: {:.2f}s".format(
                        epoch+1,
                        train_batches,
                        train_data.shape[0]/batch_size,
                        time.time()-start_time))

        training_hist.append(train_err / train_batches)

        # And a full pass over the validation data:
        print("Validating epoch...")
        val_err = 0
        val_acc = 0
        val_batches = 0
        for batch in iterate_minibatches(val_data, val_labels, batch_size, shuffle=False):
            inputs, targets = batch
            err, acc = val_fn(inputs, targets)
            val_err += err
            val_acc += acc
            val_batches += 1
            
        val_hist.append(val_err / val_batches)

        # Then we print the results for this epoch:
        print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, num_epochs, time.time() - start_time))
        print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
        print("  training accuracy:\t\t{:.2f} %".format(
            train_acc / train_batches * 100))
        print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))
        print("  validation accuracy:\t\t{:.2f} %".format(
            val_acc / val_batches * 100))

    # After training, we compute and print the test predictions/error:
    test_err = 0
    test_acc = 0
    test_batches = 0
    for batch in iterate_minibatches(test_data, test_labels, batch_size, shuffle=False):
        inputs, targets = batch
        err, acc = val_fn(inputs, targets)
        test_err += err
        test_acc += acc
        test_batches += 1
    
    test_perc = (test_acc / test_batches) * 100
    print("Final results:")
    print("  test loss:\t\t\t{:.6f}".format(test_err / test_batches))
    print("  test accuracy:\t\t{:.2f} %".format(test_perc))
        
    # Plot learning
    plt.plot(range(1, num_epochs+1), training_hist, label="Training")
    plt.plot(range(1, num_epochs+1), val_hist, label="Validation")
    plt.grid(True)
    plt.title("Training Curve\nKernel size: ({},{}) - Test acc: {:.2f}%".format(k_height, k_width, test_perc))
    plt.xlim(1, num_epochs+1)
    plt.xlabel("Epoch #")
    plt.ylabel("Loss")
    plt.legend(loc='best')
    plt.show()

    # Optionally, you could now dump the network weights to a file like this:
    # np.savez('model.npz', *lasagne.layers.get_all_param_values(network))
    #
    # And load them again later on like this:
    # with np.load('model.npz') as f:
    #     param_values = [f['arr_%d' % i] for i in range(len(f.files))]
    # lasagne.layers.set_all_param_values(network, param_values)

    return test_perc

if GRID_SEARCH:
    # Set filter sizes to search across (odd size only)
    if NUM_ELECTRODES == 4:
        search_heights = range(1, 5, 2)  # Across spatial domain (electrodes)
    elif NUM_ELECTRODES == 30:
        search_heights = range(1, 16, 4)  # Across spatial domain (electrodes)
        
    search_widths = range(1, 64, 4)  # Across temporal domain (time samples)
    
    # Preallocate accuracy grid
    grid_accuracy = np.empty((len(search_heights), len(search_widths)))
    num_kernels = grid_accuracy.size
    cur_kernel = 0
    
    for i, h in enumerate(search_heights):
        for j, w in enumerate(search_widths):
            # Train with current kernel size
            cur_kernel += 1
            print("***** Kernel {}/{} | Size: ({},{}) *****".format(cur_kernel, num_kernels, h, w))
            cur_test_acc = main(batch_size=200, num_epochs=20, k_height=h, k_width=w)
            grid_accuracy[i, j] = cur_test_acc
    
    # Show accuracy heatmap
    fig, ax = plt.subplots(figsize=(10, 10))
    heatmap = ax.imshow(grid_accuracy, cmap = plt.cm.bone, interpolation = 'mitchell')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.2)
    cb = plt.colorbar(heatmap, orientation='vertical', cax=cax)
    cb.ax.set_title('Test Acc (%)', {'fontsize': 10, 'horizontalalignment': 'left'})
    ax.grid(True)
    ax.set_xlabel('Kernel Width', weight='bold')
    ax.set_ylabel('Kernel Height', weight='bold')
    ax.xaxis.set_label_position('top')
    ax.xaxis.tick_top()
    ax.set_xticks(range(grid_accuracy.shape[1]))  # X element position
    ax.set_yticks(range(grid_accuracy.shape[0]))  # Y element position
    ax.set_xticklabels(search_widths)  # Labels for X axis
    ax.set_yticklabels(search_heights)  # Labels for Y axis
    plt.show()
    
    # Get highest accuracy and associated kernel size:
    best_idx = np.unravel_index(grid_accuracy.argmax(), grid_accuracy.shape)
    print("Highest accuracy: {:.2f}%".format(np.max(grid_accuracy)))
    print("Best kernel size: ({},{})".format(search_heights[best_idx[0]],
          search_widths[best_idx[1]]))
    
    # Highest search accuracy (4 electrodes): NA
    # Best kernel size (4 electrodes): NA
    
    # Highest search accuracy (30 electrodes): NA
    # Best kernel size (30 electrodes): NA
    
    # Train model using ideal kernel size over more epochs
    cur_test_acc = main(batch_size=200, num_epochs=300,
                        k_height=search_heights[best_idx[0]],
                        k_width=search_widths[best_idx[1]])
    
    # Final test accuracy (4 electrodes): NA
    # Final test accuracy (30 electrodes): NA
else:
    # Use best filter sizes determined from previous 4/30 electrode models
    if NUM_ELECTRODES == 4:
        cur_test_acc = main(batch_size=200, num_epochs=400, k_height=3, k_width=17)  # 66.00%
    elif NUM_ELECTRODES == 30:
        cur_test_acc = main(batch_size=200, num_epochs=400, k_height=1, k_width=25)  # 67.50%
