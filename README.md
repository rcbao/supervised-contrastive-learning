# Supervised Contrastive Learning for Brain MRI Classification

This repository implements **Supervised Contrastive Learning** for the classification of brain MRI images into three categories: **Healthy**, **Mild Cognitive Impairment (MCI)**, and **Alzheimer's Disease (AD)**. The project is based on the original Supervised Contrastive Learning framework (Khosla et al. (2021)) and has been adapted for use with a medical imaging dataset.

## Overview

- **Dataset**: Brain MRI images (grayscale, 100x76 resolution) categorized into three classes.
- **Objective**: Evaluate the performance of supervised contrastive learning for medical imaging classification.
- **Techniques Used**:
  - Data normalization with dataset-specific mean and standard deviation.
  - Handling class imbalance using a weighted sampler.
  - Experiments with data augmentation and hyperparameter tuning.

## Key Results

- Models trained on non-augmented data achieved high accuracy (~91%).
- Augmented data did not significantly improve performance and requires further investigation.
- Small batch sizes and medium-to-low learning rates yielded the best results.

## Repository Contents

- `main_supcon.py`: Implementation of supervised contrastive learning.
- `networks/resnet_big.py`: ResNet-based backbone for feature extraction.
- `data_loader.py`: Dataset processing pipeline for grayscale MRI images.
- `eval.py`: Evaluation script for generating confusion matrices, classification reports, and loss graphs.

## Setup

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/rcbao/cs-6501-mlia-supcon.git
   cd cs-6501-mlia-supcon
   ```

2. **Install Dependencies**:
   Ensure Python 3.7+ and PyTorch are installed. Use the provided `requirements.txt` file:
   ```bash
   pip install -r requirements.txt
   ```

3. **Dataset**:
   Place the brain MRI dataset in the `Classification_AD_CN_MCI_datasets/` directory. Ensure it contains the following files:
   - `brain_train_image_final.npy`
   - `brain_train_label.npy`
   - `brain_test_image_final.npy`
   - `brain_test_label.npy`

## How to run the code

   To train the model and generate metrics, loss curves, and confusion matrices for a combination of parameters:
   ```bash
   python eval.py
   ```

  The entire training run will take two to three hours to complete.

## Future Work

- Investigate more effective data augmentation techniques for medical imaging.
- Explore longer training runs with increased epochs for improved results.
- Optimize hyperparameter tuning for further performance gains.
