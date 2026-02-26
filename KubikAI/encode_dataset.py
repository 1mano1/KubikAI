import os
import sys
import json
import argparse
import torch
import numpy as np
from glob import glob
from tqdm import tqdm

# Add project root to path to allow absolute imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from KubikAI.models.sdf_vae import SdfVAE

@torch.no_grad()
def encode_dataset(args):
    """
    Loads a trained SdfVAE and uses its encoder to convert a dataset of SDF point clouds
    into a dataset of latent vectors.
    """
    print("Loading VAE model...")
    # We instantiate the model with the same architecture as during training
    model = SdfVAE(latent_dim=256, num_points=16384, resolution=16)
    
    # Load the checkpoint from the VAE training
    state_dict = torch.load(args.vae_ckpt, map_location='cpu')
    model.load_state_dict(state_dict)
    model.eval().cuda()
    print("Model loaded successfully.")

    data_roots = [path.strip() for path in args.data_dir.split(',')]
    
    print("Finding SDF files...")
    sdf_files = []
    for root in data_roots:
        # We assume the structure `.../{dataset_name}/{model_name}/sdf/samples.npz`
        nested_files = glob(os.path.join(root, '**', 'sdf', 'samples.npz'), recursive=True)
        sdf_files.extend(nested_files)
    print(f"Found {len(sdf_files)} files to encode.")

    for sdf_path in tqdm(sdf_files, desc="Encoding dataset"):
        try:
            data = np.load(sdf_path)
            points = torch.from_numpy(data['points']).float().cuda()
            sdf = torch.from_numpy(data['sdf']).float().cuda()
            
            if len(sdf.shape) == 1:
                sdf = sdf[:, None]

            # The VAE encoder expects a batch dimension, so we add one
            points = points.unsqueeze(0)
            sdf = sdf.unsqueeze(0)

            # The SDF file can have many points, but the VAE was trained on a fixed number.
            # We must subsample the points to match the model's expected input.
            if points.shape[1] > model.encoder.num_points:
                indices = torch.randperm(points.shape[1])[:model.encoder.num_points]
                points = points[:, indices, :]
                sdf = sdf[:, indices, :]
            
            # --- Encode the SDF data to get the latent vector ---
            mean, _ = model.encoder(points, sdf)
            
            # --- Construct the output path to mirror the input structure ---
            # e.g., input:  ./KubikAI/processed_datasets/Fornite/Barbosa/sdf/samples.npz
            #       output: ./KubikAI/encoded_datasets/Fornite/Barbosa/latent.pt
            
            # The path is split and reconstructed to create the output directory
            # This assumes a consistent structure.
            parts = sdf_path.split(os.path.sep)
            model_name = parts[-3]
            dataset_name = parts[-4]
            
            out_dir = os.path.join(args.output_dir, dataset_name, model_name)
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, 'latent.pt')
            
            # Save the latent vector (the mean of the distribution) to CPU to make it portable
            torch.save(mean.squeeze(0).cpu(), out_path)

        except Exception as e:
            print(f"Failed to process {sdf_path}: {e}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Encode a dataset using a trained SdfVAE.")
    parser.add_argument('--vae_ckpt', type=str, required=True, help='Path to the trained SdfVAE checkpoint (e.g., outputs/sdf_vae_training/ckpts/vae_step0500000.pt)')
    parser.add_argument('--data_dir', type=str, required=True, help='Comma-separated paths to the ROOT of processed SDF datasets.')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save the encoded latent vectors.')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.vae_ckpt):
        raise FileNotFoundError(f"VAE checkpoint not found at: {args.vae_ckpt}")
        
    encode_dataset(args)
    print("Dataset encoding complete.")
