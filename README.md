# Patch-Wise: Image Inpainting via Convolutional Autoencoders

Patch-Wise is a deep learning project that reconstructs corrupted images using a Convolutional Autoencoder built with PyTorch. It artificially corrupts training images by applying random black patches and trains a neural network to predict and fill in the missing spatial information (image inpainting), restoring the image to its original state.

## Project Overview

This repository demonstrates the ability of convolutional networks to learn spatial hierarchies and context. By mapping heavily corrupted inputs (with randomized missing patches) to clean ground-truth images, the model learns the underlying distribution of the image dataset (in this case, cars) and successfully performs denoising and inpainting.

### Key Features
* **Custom Patching Transform:** A dynamic `randomPatchify` class that generates random sizes and numbers of black patches on the fly during the dataloader phase.
* **Fully Convolutional Autoencoder:** An encoder-decoder architecture utilizing `Conv2d` for feature extraction and dimensionality reduction, and `ConvTranspose2d` for upsampling and reconstruction.
* **End-to-End Pipeline:** Complete workflow from data loading and augmentation to training, validation, and visual evaluation.

## Repository Structure

```text
patch-wise/
├── data/
│   └── cars/               # Directory containing the dataset (.jpg, .png)
├── notebooks/
│   └── main.ipynb          # Jupyter notebook with the full pipeline and visualizations
├── src/
│   ├── dataset.py          # Custom PyTorch Dataset and Patching Transforms
│   ├── model.py            # ConvAutoencoder architecture
│   └── utils.py            # Utility functions for patching and visualization
├── requirements.txt        # Project dependencies
└── README.md