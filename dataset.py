# ===== File name: ./dataset.py =====

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

class BrainImageDataset(Dataset):
    def __init__(self, image_path, label_path, transform=None):
        """
        Args:
            image_path (str): Path to the .npy file with images.
            label_path (str): Path to the .npy file with labels.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.images = np.load(image_path)
        self.labels = np.load(label_path)
        self.transform = transform

        # Verify image dimensions
        assert self.images.ndim == 4, f"Expected 4D images, got {self.images.ndim}D"
        assert self.images.shape[1] == 2, f"Expected 2 channels (template & input), got {self.images.shape[1]}"

        # Select the input image (channel index 1)
        self.images = self.images[:, 1, :, :]  # Shape: (num_samples, 100, 76)

        # Expand channels to make it compatible with RGB (required by torchvision transforms)
        self.images = np.expand_dims(self.images, axis=-1)  # Shape: (num_samples, 100, 76, 1)
        self.images = np.repeat(self.images, 3, axis=-1)   # Shape: (num_samples, 100, 76, 3)

        # Normalize if images are in [0, 255]
        if self.images.max() > 1.0:
            self.images = self.images.astype('float32') / 255.0

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        # Convert numpy array to PIL Image
        image = Image.fromarray((image * 255).astype('uint8'), 'RGB')

        if self.transform:
            image = self.transform(image)

        return image, label
