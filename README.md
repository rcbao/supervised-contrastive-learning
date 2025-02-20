# Using Supervised Contrastive Learning to Classify Brain MRI Images

<p align="center">
  <img src="https://github.com/user-attachments/assets/ccfb2bc3-cdfb-415f-883e-9a544c4e2acc" width="48%" />
  <img src="https://github.com/user-attachments/assets/4b8459e4-fc9f-4a5b-88de-03316351c76e" width="48%" />
</p>

This project applys **Supervised Contrastive Learning** to classify brain MRI images into three categories: **Healthy**, **Mild Cognitive Impairment (MCI)**, and **Alzheimer's Disease (AD)**. 

The project is based on the Supervised Contrastive Learning framework proposed by [Khosla et al.](https://arxiv.org/pdf/2004.11362). The code has been adapted for pre-training and inference on the brain MRI dataset.

Read our [full project report](REPORT.pdf) for more details.

## Overview

- **Dataset**: Brain MRI images (grayscale, 100x76 resolution) categorized into three classes.
- **Objective**: Evaluate the performance of supervised contrastive learning for medical imaging classification.
- **Techniques Used**:
  - Data normalization with dataset-specific mean and standard deviation.
  - Handling class imbalance using a weighted sampler.
  - Experiments with data augmentation and hyperparameter tuning.

## Selected Results

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

## How to Run the Code

To train the model and generate metrics, loss curves, and confusion matrices for different parameter combinations, run:

```bash
python eval_v1.py
```

The complete training process, including evaluation, will take approximately 2–3 hours, depending on the hardware configuration.

## Pre-trained Model

The best-performing pre-trained model, achieved with a batch size of 32, learning rate of 0.01, and no data augmentation, is provided as `batchsize-32-lr-0.01-transform-base.pth.zip`.

To use the model:
1. Unzip the file:
   ```bash
   unzip batchsize-32-lr-0.01-transform-base.pth.zip
   ```
2. Load the model for evaluation or inference as needed.

## Future Work

- Investigate more effective data augmentation techniques for medical imaging.
- Explore longer training runs with increased epochs for improved results.
- Optimize hyperparameter tuning for further performance gains.
