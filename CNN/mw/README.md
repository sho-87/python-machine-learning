# EEG Mind Wandering

Classification of EEG data using Lasagne

* **1_eeg_mw_2d.py**
  * 2D convolutions. Consumes a large amount of memory and very slow

* **2_eeg_mw_1d.py**
  * Changed to 1D convolution for speed and memory

* **3_eeg_mw_sd.py**
  * Data standardized to help training

* **4_eeg_mw_dropout.py**
  * Dropout added to prevent overfitting

* **5a_eeg_mw_augment_linear.py**
  * Augment data with linear transformations (scale and translation) to get more training examples
