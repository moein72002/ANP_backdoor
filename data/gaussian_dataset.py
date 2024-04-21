import torch
from torch.utils.data import Dataset
import numpy as np

class GaussianNoiseDataset(Dataset):
    def __init__(self, image_size, length, mean=0, std=1):
        """
        Args:
            image_size (tuple of int): The dimensions of the images (height, width).
            length (int): The total number of images in the dataset.
            mean (float): The mean of the Gaussian noise distribution.
            std (float): The standard deviation of the Gaussian noise distribution.
        """
        self.length = length
        self.image_size = image_size
        self.mean = mean
        self.std = std

        # Generate all noise images in advance
        self.noise_images = [self.generate_noise_image() for _ in range(length)]

    def generate_noise_image(self):
        """Generate a single noise image."""
        noise_image = np.random.normal(self.mean, self.std, self.image_size)
        noise_image = torch.from_numpy(noise_image).float()
        return noise_image

    def __len__(self):
        """
        Returns:
            int: The total number of images in the dataset.
        """
        return self.length

    def __getitem__(self, index):
        """
        Args:
            index (int): The index of the item to return.

        Returns:
            torch.Tensor: An image-sized tensor filled with Gaussian noise.
        """
        return self.noise_images[index], 0 # label is None

