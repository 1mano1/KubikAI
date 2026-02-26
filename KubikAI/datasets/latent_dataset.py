import os
import torch
from torch.utils.data import Dataset
import numpy as np
from glob import glob
from PIL import Image
from torchvision import transforms

class LatentImageDataset(Dataset):
    """
    A dataset that loads pre-computed latent vectors and their corresponding images.
    
    This dataset is used to train the Flow model. It pairs a latent vector 'z' (encoded by the VAE)
    with a conditioning image that represents the 3D model.

    Assumes a directory structure where:
    - Latents are in: `encoded_root/{dataset_name}/{model_name}/latent.pt`
    - Images are in: `processed_root/{dataset_name}/{model_name}/renders/000.png` (using the first render as default)
    """

    def __init__(self, encoded_root, processed_root, resolution=224):
        """
        Args:
            encoded_root (str): The root directory of the encoded latent vectors.
            processed_root (str): The root directory of the SDF datasets, which contains the render images.
            resolution (int): The resolution to resize images to.
        """
        self.latent_files = glob(os.path.join(encoded_root, '**', 'latent.pt'), recursive=True)
        self.processed_root = processed_root
        
        self.transform = transforms.Compose([
            transforms.Resize((resolution, resolution)),
            transforms.ToTensor(),
        ])
        
        print(f"Found {len(self.latent_files)} latent vector files.")

    def __len__(self):
        return len(self.latent_files)

    def __getitem__(self, index):
        latent_path = self.latent_files[index]
        
        # Load the pre-computed latent vector
        latent = torch.load(latent_path)

        # Construct the corresponding image path from the latent path
        # e.g., .../encoded_datasets/Fornite/Barbosa/latent.pt
        # -> .../processed_datasets/Fornite/Barbosa/renders/000.png
        parts = latent_path.split(os.path.sep)
        model_name = parts[-2]
        dataset_name = parts[-3]
        
        image_path = os.path.join(self.processed_root, dataset_name, model_name, 'renders', '000.png')
        
        try:
            image = Image.open(image_path).convert('RGB')
            image = self.transform(image)
        except Exception as e:
            print(f"Error loading image {image_path}, skipping: {e}")
            # Return the next valid sample to prevent a training crash
            return self.__getitem__((index + 1) % len(self))

        return {
            'latent': latent,  # This will be x_0 for the Flow model
            'image': image   # This will be the conditioning signal
        }


def collate_fn(batch):
    """
    Custom collate function for the LatentImageDataset.
    """
    batch = [item for item in batch if item is not None]
    if not batch:
        return None

    latents = torch.stack([item['latent'] for item in batch])
    images = torch.stack([item['image'] for item in batch])
    
    return {
        'latent': latents,
        'image': images
    }
