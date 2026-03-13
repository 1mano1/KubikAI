import os
import torch
from torch.utils.data import Dataset
import numpy as np
from glob import glob

class SdfDataset(Dataset):
    """
    A flexible dataset for loading pre-computed Signed Distance Function (SDF) samples.
    It can handle multiple data sources, each with a different directory structure.
    
    It supports two main structures:
    1. Nested Structure (like our processed Fornite/Anime datasets):
        /root/
            - ModelNameA/
                - sdf/
                    - samples.npz
            - ModelNameB/
                - sdf/
                    - samples.npz
                    
    2. Flat Structure (like Objaverse):
        /root/
            - sdf_samples/
                - hash1.npz
                - hash2.npz
    """

    def __init__(self, roots, num_samples=16384):
        """
        Initializes the dataset.
        
        Args:
            roots (list): A list of root directories for the datasets.
            num_samples (int): The number of points to subsample from each SDF file.
        """
        self.roots = roots if isinstance(roots, list) else [roots]
        self.num_samples = num_samples
        self.sdf_files = self._find_all_sdf_files()

    def _find_all_sdf_files(self):
        all_files = []
        for root in self.roots:
            # Look for nested structure (our custom format)
            nested_files = glob(os.path.join(root, '**', 'sdf', 'samples.npz'), recursive=True)
            all_files.extend(nested_files)

            # Look for flat structure (Objaverse format)
            flat_files = glob(os.path.join(root, 'sdf_samples', '*.npz'), recursive=True)
            all_files.extend(flat_files)

            # Look for *.npz files recursively in any subfolder (Kaggle processing format)
            recursive_npz = glob(os.path.join(root, '**', '*.npz'), recursive=True)
            all_files.extend(recursive_npz)
        
        # Remove duplicates while preserving order
        all_files = list(dict.fromkeys(all_files))
        
        if not all_files:
            raise ValueError(f"No SDF files found in the provided root directories: {self.roots}")
            
        print(f"Found {len(all_files)} SDF files across {len(self.roots)} data sources.")
        return all_files

    def __len__(self):
        return len(self.sdf_files)

    def __getitem__(self, index):
        sdf_path = self.sdf_files[index]
        
        try:
            data = np.load(sdf_path)
            points = data['points']
            sdf = data['sdf']

            # Ensure sdf is in the correct shape (N, 1)
            if len(sdf.shape) == 1:
                sdf = sdf[:, np.newaxis]

            # The stored file might have more samples than we need for training
            # So we subsample them. This also ensures consistent sample sizes.
            if points.shape[0] < self.num_samples:
                # If the file has fewer samples, we sample with replacement
                indices = np.random.choice(points.shape[0], self.num_samples, replace=True)
            else:
                # If the file has enough samples, we sample without replacement
                indices = np.random.choice(points.shape[0], self.num_samples, replace=False)
            
            points_subset = torch.from_numpy(points[indices, :]).float()
            sdf_subset = torch.from_numpy(sdf[indices, :]).float()

            # Clamp SDF values to focus learning on the surface boundary
            clamp_val = 0.1
            sdf_subset = torch.clamp(sdf_subset, -clamp_val, clamp_val)

            return {
                'points': points_subset,
                'sdf': sdf_subset
            }
        except Exception as e:
            print(f"Error loading or processing file: {sdf_path}")
            # Return the next valid sample to prevent training crash
            return self.__getitem__((index + 1) % len(self))


def collate_fn(batch):
    """
    Custom collate function for the SdfDataset.
    """
    # Filters out any samples that might have failed to load
    batch = [item for item in batch if item is not None]
    if not batch:
        return None

    points = torch.stack([item['points'] for item in batch])
    sdf = torch.stack([item['sdf'] for item in batch])
    return {
        'points': points,
        'sdf': sdf
    }
