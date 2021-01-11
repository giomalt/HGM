# HGM

Programs and neural network used to perform **Hermite Gaussian Microscopy (HGM)** as presented in the paper **"Super-resolution linear optical microscopy in the far field"**.

1. **main_processing.ipynb**

Generate sources, simulate HGM and direct imaging, create the images and sequences for the Digital Micromirror Device (DMD), process experimental oscilloscope data to produce the tensors to feed the neural network (NN) input.

2. **acquisition_DMD_pattern_on_fly.ipynb**

Implement HGM in the lab.
Drive DMD, oscilloscope and power spectrum analyzer and includes the automatic interferometer alignment procedure.

3. **neural_network/model.pynb**

Train the NN to learn HGM underlying model from the simulated and acquired experimental data.
