# EEG Mind Wandering

Classification of EEG data using Lasagne

* **1_eeg_mw_2d.py**
  * 2D convolutions

* **1b_eeg_mw_1d.py**
  * 1D convolutions

* **2_eeg_mw_sd.py**
  * Data standardized to help gradient descent

* **3_eeg_mw_regularize.py**
  * Dropout and weight decay added to help overfitting

* **4_eeg_mw_augment.py**
  * Augment training data with scale/translation

* **5_eeg_mw_search_kernel.py**
  * Search for ideal kernel size. Few epochs during search (time intensive) - increased learning rate to compensate. Early stopping introduced

* **6_eeg_mw_electrodes.py**
  * Down sample # of electrodes to 30, and reorder them spatially so kernels can learn spatial relationships

* **7_eeg_mw_electrodes_downsample.py**
  * Further down sample electrodes to only the (4) specific electrodes that have been shown to be important in the literature

* **8_eeg_mw_bands.py**
  * Extract frequency bands (alpha, delta, beta etc.) and use them as additional channels

* **9_eeg_mw_xcorr.py**
  * Calculate cross-correlation matrix across all electrodes (and frequency bands) and use xcorr as the input into CNN
