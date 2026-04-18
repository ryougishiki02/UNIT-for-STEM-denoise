# UNIT-for-STEM-denoise

This project provides an unsupervised deep learning framework based on the UNIT architecture to denoise high-resolution Scanning Transmission Electron Microscopy (STEM) images. By integrating both real-space and reciprocal-space (Fourier) loss functions, our method successfully suppresses complex noises while strictly preserving long-range lattice periodicity and local defect features.

---

## Features

- ✅ **Unsupervised Learning**: Trains on unpaired simulated images and noisy experimental data, eliminating the need for noise-free ground truth.
- ✅ **Dual-Domain Loss**: Balances structural details and periodicity via spatial and frequency-domain constraints.
- ✅ **Robust Generalization**: Validated on various 2D materials including MoS2, SnSe, and Cu2Te.
- ✅ **Artifact-Free Reconstruction**: Employs an overlapping patch segmentation and padding strategy to process large-scale images without boundary artifacts.

---

## Installation

Clone the repository:

```bash
git clone [https://github.com/ryougishiki02/UNIT-for-STEM-denoise.git](https://github.com/ryougishiki02/UNIT-for-STEM-denoise.git)
cd UNIT-for-STEM-denoise
