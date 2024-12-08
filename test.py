import numpy as np
import torch
from torchvision import transforms
import matplotlib.pyplot as plt
import os

# Create a directory to save plots
os.makedirs("debug_plots", exist_ok=True)

# Load dataset
train_images = np.load('Classification_AD_CN_MCI_datasets/brain_train_image_final.npy')
train_labels = np.load('Classification_AD_CN_MCI_datasets/brain_train_label.npy')

# Verify dataset shape
print(f"Training images shape: {train_images.shape}")
print(f"Training labels shape: {train_labels.shape}")

# Check unique labels
unique_labels, counts = np.unique(train_labels, return_counts=True)
print(f"Unique labels: {unique_labels}, Counts: {counts}")

# Extract the input channel as specified
train_images_modified = train_images[:, 1, :, :]
print(f"Modified training images shape: {train_images_modified.shape}")

# Check pixel range
print(f"Pixel range before normalization: Min={np.min(train_images_modified)}, Max={np.max(train_images_modified)}")

# Apply normalization transform
normalize_transform = transforms.Normalize(mean=[0.00143], std=[0.00149])

# Add channel dimension to match (C, H, W) format
sample_tensor = torch.tensor(train_images_modified[0], dtype=torch.float32).unsqueeze(0)  # Shape: (1, 100, 76)
sample_tensor = sample_tensor / 255.0  # Ensure pixel values are in [0, 1]
normalized_tensor = normalize_transform(sample_tensor)

# Check pixel range after normalization
print(f"Pixel range after normalization: Min={torch.min(normalized_tensor)}, Max={torch.max(normalized_tensor)}")

# Save sample image plots
plt.figure(figsize=(10, 5))

# Original image
plt.subplot(1, 2, 1)
plt.imshow(train_images_modified[0], cmap='gray')
plt.title("Original Image")
plt.colorbar()

# Normalized image
plt.subplot(1, 2, 2)
plt.imshow(normalized_tensor.squeeze(0).numpy(), cmap='gray')  # Remove channel dimension for display
plt.title("Normalized Image")
plt.colorbar()

plt.savefig("debug_plots/normalization_check.png")
plt.close()

# Test augmentations
augmentations = transforms.Compose([
    transforms.RandomResizedCrop((80, 60)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
])



# Check DataLoader batch and labels
from torch.utils.data import DataLoader, TensorDataset

# Create Dataset and DataLoader
dataset = TensorDataset(torch.tensor(train_images_modified, dtype=torch.float32).unsqueeze(1),  # Add channel dimension
                        torch.tensor(train_labels))
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Check first batch
for batch_idx, (images, labels) in enumerate(dataloader):
    print(f"Batch {batch_idx}: Images Shape={images.shape}, Labels Shape={labels.shape}")
    print(f"First 5 labels in batch: {labels[:5]}")
    break

# Check gradient updates
from torch.autograd import Variable
import torch.nn as nn

# Dummy model for gradient check
model = nn.Sequential(nn.Flatten(), nn.Linear(80 * 60, 3))  # Simple linear classifier
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)


# Save gradient norms to a text file
with open("debug_plots/gradient_check.txt", "w") as grad_file:
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_file.write(f"{name}: gradient norm = {param.grad.norm().item()}\n")
