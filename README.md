# LSM_SpiNNaker
 LSM trained in PyTorch and implemented on SpiNNaker for the N-MNIST Dataset
 ![g](lsm.png)
 
 
 
#### Usage: 
- Pytorch training (readout layer): pytorch/LSM_NMNIST.ipynb
- SpiNNaker inference: spinnaker/spinnaker_inference.ipynb

#### Prerequisites: 
- NMNIST Dataset. Here we use the 50 frames, downloadable [here](https://drive.google.com/drive/folders/1XkqIHMioy4fmbTgJ__x4bjN_YWs9XU4a?usp=sharing).
- pytorch
- matplotlib
- h5py
- spynnaker. Tested on SpiNN-3, SpiNN-5 and HBP Server

#### Citation
- Patiño-Saucedo, Alberto, et al. "Liquid State Machine on SpiNNaker for Spatio-Temporal Classification Tasks." Frontiers in Neuroscience 16 (2022).
