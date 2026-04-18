# UNIT for STEM Image Denoising

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
![Python 3.8+](https://img.shields.io/badge/python-3.8+-green.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-red.svg)

Official implementation of the paper **"Unsupervised Denoising of STEM Images for Enhanced Atomic Quantification"** (published in *Ultramicroscopy*).

In this tutorial, we will guide you through setting up the environment for running our UNIT-based denoising framework and show how to apply it to your high-resolution Scanning Transmission Electron Microscopy (STEM) images.

## Background & Algorithm

Traditional STEM image denoising methods often struggle to balance the preservation of local defect features with global lattice periodicity. Our method approaches this by treating denoising as an **unsupervised image-to-image translation** problem.

We utilize the Unsupervised Image-to-Image Translation (UNIT) architecture. The training data consists of two unpaired datasets:
1. **Source Domain:** Noisy experimental STEM images.
2. **Target Domain:** Noise-free simulated STEM images (e.g., from TEM-ImageNet).

By leveraging a shared-latent space assumption and incorporating both **real-space and reciprocal-space (Fourier) loss functions**, the network learns to suppress complex shot and scan noises while strictly preserving atomic positions and local defect features.

## Requirements

- **Hardware:** PC with an NVIDIA GPU (e.g., RTX 3090, Tesla V100) with at least 8GB+ GPU memory. 
- **Software:** *Ubuntu / Windows*, *CUDA 11.x*, *Anaconda3*
- **Python Packages:**
  ```bash
  conda create -n stem_denoise python=3.8
  conda activate stem_denoise
  conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
  pip install numpy opencv-python matplotlib
