# EEG Mind Wandering

Classification of EEG data using Lasagne

* **1_eeg_mw_2d.py**
  * 2D convolutions

* **2_eeg_mw_sd.py**
  * Data standardized to help training

* **3_eeg_mw_regularize.py**
  * Dropout and weight decay added to help overfitting

* **4_eeg_mw_electrodes.py**
  * Down sample # of electrodes to 30, and reorder them spatially so kernels can learn spatial relationships

* **5_eeg_mw_bands.py**
  * Extract frequency bands (alpha, delta, beta etc.) and use them as additional channels
