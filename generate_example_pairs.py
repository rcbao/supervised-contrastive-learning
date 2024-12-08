import os
import numpy as np
import torch
from PIL import Image
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend
import matplotlib.pyplot as plt
from collections import defaultdict
import random

# Directory to save example pairs
os.makedirs("example_pairs", exist_ok=True)

# Load your dataset (.npy files)
train_images = np.load('Classification_AD_CN_MCI_datasets/brain_train_image_final.npy')
train_labels = np.load('Classification_AD_CN_MCI_datasets/brain_train_label.npy')

# Modify images (discard first channel)
train_images_mod = train_images[:, 1, :, :]  # Shape: (num_samples, 100, 76)

# Group indices by label
label_to_indices = defaultdict(list)
for idx, label in enumerate(train_labels):
    label_to_indices[label].append(idx)

# Ensure there are enough samples per class
for label, indices in label_to_indices.items():
    if len(indices) < 2:
        raise ValueError(f"Not enough samples for label {label} to form a positive pair.")

# Function to de-normalize and convert to PIL Image
def fetch_image(images, idx):
    img = images[idx]  # Shape: (100, 76)
    img = np.clip(img, 0, 1)  # Assuming images are already scaled to [0,1]
    img = (img * 255).astype(np.uint8)  # Scale to [0,255]
    img_pil = Image.fromarray(img, mode='L')  # 'L' mode for single-channel
    return img_pil

# Select a positive pair
def get_positive_pair():
    chosen_label = random.choice(list(label_to_indices.keys()))
    pos_indices = random.sample(label_to_indices[chosen_label], 2)
    pos_img1 = fetch_image(train_images_mod, pos_indices[0])
    pos_img2 = fetch_image(train_images_mod, pos_indices[1])
    return chosen_label, pos_img1, pos_img2

# Select a negative pair
def get_negative_pair():
    label1, label2 = random.sample(list(label_to_indices.keys()), 2)
    neg_idx1 = random.choice(label_to_indices[label1])
    neg_idx2 = random.choice(label_to_indices[label2])
    neg_img1 = fetch_image(train_images_mod, neg_idx1)
    neg_img2 = fetch_image(train_images_mod, neg_idx2)
    return (label1, label2), neg_img1, neg_img2

# Save the pairs as images
def save_pairs():
    # Positive Pair
    pos_label, pos_img1, pos_img2 = get_positive_pair()
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(pos_img1, cmap='gray')
    axes[0].set_title(f"Positive Pair - Label: {pos_label}")
    axes[0].axis('off')
    axes[1].imshow(pos_img2, cmap='gray')
    axes[1].set_title(f"Positive Pair - Label: {pos_label}")
    axes[1].axis('off')
    plt.tight_layout()
    plt.savefig(f"example_pairs/positive-pair-label-{pos_label}.png")
    plt.close()

    # Negative Pair
    (neg_label1, neg_label2), neg_img1, neg_img2 = get_negative_pair()
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(neg_img1, cmap='gray')
    axes[0].set_title(f"Negative Pair - Label: {neg_label1}")
    axes[0].axis('off')
    axes[1].imshow(neg_img2, cmap='gray')
    axes[1].set_title(f"Negative Pair - Label: {neg_label2}")
    axes[1].axis('off')
    plt.tight_layout()
    plt.savefig(f"example_pairs/negative-pair-labels-{neg_label1}-{neg_label2}.png")
    plt.close()

    print("Saved positive and negative pairs to the 'example_pairs/' directory.")

if __name__ == "__main__":
    save_pairs()
