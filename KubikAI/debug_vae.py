import os
import sys
import torch
import numpy as np
import trimesh
from skimage import measure

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from KubikAI.models.sdf_vae import SdfVAE

@torch.no_grad()
def decode_to_mesh(vae, latent, resolution=64, threshold=0.0, device='cuda'):
    """
    Decodes a latent grid to a mesh using Marching Cubes.
    (Simplified from inference.py)
    """
    grid_points = np.linspace(-1, 1, resolution)
    grid_points = np.stack(np.meshgrid(grid_points, grid_points, grid_points, indexing='ij'), axis=-1)
    grid_points = grid_points.reshape(-1, 3)
    
    chunk_size = 100000
    sdf_values = []
    points_tensor = torch.from_numpy(grid_points).float().to(device)
    
    for i in range(0, points_tensor.shape[0], chunk_size):
        chunk = points_tensor[i:i+chunk_size].unsqueeze(0)
        sdf_out = vae.decoder(latent, chunk)
        sdf_values.append(sdf_out.squeeze(0).cpu().numpy())
        
    sdf_values = np.concatenate(sdf_values, axis=0)
    sdf_volume = sdf_values.reshape(resolution, resolution, resolution)
    
    print(f"SDF Volume Stats: Min={sdf_volume.min():.4f}, Max={sdf_volume.max():.4f}, Mean={sdf_volume.mean():.4f}")

    if sdf_volume.min() > threshold or sdf_volume.max() < threshold:
        print("SDF volume does not cross the threshold. No surface found.")
        return None
        
    verts, faces, normals, values = measure.marching_cubes(sdf_volume, level=threshold)
    verts = verts / (resolution - 1) * 2 - 1
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=normals)
    return mesh

def main(vae_ckpt):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    print("Loading VAE model...")
    try:
        vae = SdfVAE(latent_dim=256, num_points=16384, resolution=16)
        state_dict = torch.load(vae_ckpt, map_location='cpu')
        vae.load_state_dict(state_dict)
        vae.eval().to(device)
        print("VAE model loaded successfully.")
    except Exception as e:
        print(f"Error loading VAE: {e}")
        return

    # Create a synthetic latent vector (a grid of zeros)
    # This tests what the model has learned in its biases.
    print("Creating a synthetic latent grid (all zeros)...")
    latent_shape = (1, 256, 16, 16, 16)
    synthetic_latent = torch.zeros(latent_shape, device=device)

    print("Attempting to decode synthetic latent...")
    mesh = decode_to_mesh(vae, synthetic_latent, device=device)

    if mesh:
        output_path = "/kaggle/working/debug_shape.obj"
        mesh.export(output_path)
        print(f"SUCCESS: Generated a debug shape and saved to {output_path}")
    else:
        print("FAILURE: Could not generate a mesh from the VAE decoder.")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--vae_ckpt', type=str, required=True, help='Path to VAE checkpoint')
    args = parser.parse_args()
    main(args.vae_ckpt)
