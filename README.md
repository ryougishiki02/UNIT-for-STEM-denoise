# UNIT for STEM Image Denoising

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
![Python 3.10](https://img.shields.io/badge/python-3.10-green.svg)
![PyTorch 2.2.2](https://img.shields.io/badge/PyTorch-2.2.2-red.svg)

Official implementation of the paper **"Unsupervised Denoising of STEM Images for Enhanced Atomic Quantification"** (published in *Ultramicroscopy*).

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
```

---

## Recommended Environment

It is recommended to use the following environment (via `conda`) to ensure compatibility:

- **Python**: 3.10 
- **PyTorch**: 2.2.2 + CUDA 11.8  
- **NumPy**: 1.26 (⚠️ PyTorch 2.2.2 is incompatible with NumPy ≥ 2.0)  
- **OpenCV**: 4.8.1  
- **Others**:  
  - SciPy 1.11  
  - scikit-learn 1.3  
  - pandas  
  - scikit-image  
  - matplotlib  

---

## Directory Structure

Before testing, please organize your noisy STEM images. Place the images you want to denoise into the `divide/inputs/` directory:

```text
UNIT-for-STEM-denoise/
├── divide/
│   ├── inputs/       <-- Put your noisy experimental images here (.jpg, .png, .tif)
│   └── divide.py     <-- Script for patch segmentation
├── combine/
│   └── outputs/      <-- Denoised full-size results will appear here
├── all.py            <-- Main execution script
└── checkpoints/      <-- Pre-trained model weights
```

---

## Pretrained Models

Download our pretrained models and put them in the `checkpoints/` folder.

| Dataset / Material | Model Link |
|--------------------|------------|
| Universal 2D STEM  | [Insert Download Link Here] | 

---

## Usage

For high-resolution STEM images, directly feeding them into the network can cause memory issues and boundary artifacts. We employ an overlapping patch segmentation and padding strategy.

**Step 1: Configure Segmentation**
Open `divide/divide.py` to set your patch segmentation parameters:
- `n`: Determines the grid size for cropping ($n^2$ total patches).
- `pixel_increase`: Defines the expanded pixel size for each patch to ensure overlapping boundaries.

**Step 2: Run Denoising**
Execute the main script from your terminal:
```bash
python all.py
```

**Step 3: View Results**
The final, seamlessly stitched denoised images will be saved in `combine/outputs/`. You should see results similar to the following:

| Material | Noisy Input | Denoised Output |
|----------|-------------|-----------------|
| **MoS2** | <img src="./assets/mos2_input.jpg" width="384" title="Noisy MoS2"> | <img src="./assets/mos2_output.jpg" width="384" title="Denoised MoS2"> |
| **SnSe** | <img src="./assets/snse_input.jpg" width="384" title="Noisy SnSe"> | <img src="./assets/snse_output.jpg" width="384" title="Denoised SnSe"> |

*(Note: Create an `assets` folder and upload your example images to render the comparison table properly).*

---

## Citation

If you use any code or models from this repository, please cite our paper:

```bibtex
@article{wu2024unsupervised,
  title={Unsupervised Denoising of STEM Images for Enhanced Atomic Quantification},
  author={Wu, Yucong and Huang, Ziyang and Chen, Honglue and He, Bohua and Qi, Meng and Zheng, He and Zhao, Peili and Jia, Shuangfeng and Wang, Jianbo},
  journal={Ultramicroscopy},
  year={2024},
  doi={[Insert Your DOI Here]}
}
```

---

## Contact

For questions or contributions, please contact:  
**Yucong Wu** (wuyucong@whu.edu.cn)  
**Ziyang Huang** (huangziyang02@gmail.com)
