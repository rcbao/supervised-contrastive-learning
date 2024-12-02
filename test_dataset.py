import numpy as np
from dataset import BrainImageDataset
import matplotlib.pyplot as plt
from torchvision import transforms

# Load dataset
train_dataset = BrainImageDataset(
    image_path='./Classification_AD_CN_MCI_datasets/brain_train_image_final.npy',
    label_path='./Classification_AD_CN_MCI_datasets/brain_train_label.npy',
    transform=None
)

# Print shapes
print(f"Training images shape: {train_dataset.images.shape}")  # Should be (1657, 100, 76, 3)
print(f"Training labels shape: {train_dataset.labels.shape}")  # Should be (1657,)

# Visualize a sample
image, label = train_dataset[0]
transform = transforms.ToTensor()
image = transform(image)

plt.imshow(image.permute(1, 2, 0))
filename = f'image_{label}.png'
plt.savefig(filename)
plt.close()