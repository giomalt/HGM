# HGM
Programs and neural network (NN) used to perform **Hermite Gaussian Microscopy (HGM)**, as presented in the paper **"Super-resolution linear optical microscopy in the far field"**.

**Main programs**
1. ```main_processing.ipynb```   
A) Generate sources (train/dev/test datasets)   
B) Simulate HGM reconstructions and camera images   
C) Create images and sequences for the Digital Micromirror Device (DMD)      
D) Process experimental oscilloscope data to produce the tensors which are inputs to the NN   

2. ```experiment/acquisition_DMD_pattern_on_fly.ipynb```   
Procedure to experimentally perform HGM: drive the DMD in pattern-on-the-fly mode, read oscilloscope and power spectrum analyzer data, periodically perform the automatic interferometer alignment procedure.

3. ```neural_network/model.pynb```   
Deep learning model to map the acquired experimental photocurrents to the simulated HGM reconstruction images.


**Other programs**   
```experiment/acquire_camera_images.ipynb```: automatically acquire camera images of the test dataset by performing a scan acquisition

```deconvolution/deconvolution.ipynb```: deconvolve camera images via Richardson-Lucy (RL) algorithm
