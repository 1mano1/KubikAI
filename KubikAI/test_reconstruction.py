import os
import sys
import torch
import numpy as np
import trimesh
import argparse
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
    """
    grid_points = np.linspace(-1, 1, resolution)
    grid_points = np.stack(np.meshgrid(grid_points, grid_points, grid_points, indexing='ij'), axis=-1)
    grid_points = grid_points.reshape(-1, 3)
    
    chunk_size = 50000  # Reduced chunk size to prevent OOM
    sdf_values = []
    points_tensor = torch.from_numpy(grid_points).float().to(device)
    
    print(f"Decoding grid with resolution {resolution}...")
    for i in range(0, points_tensor.shape[0], chunk_size):
        chunk = points_tensor[i:i+chunk_size].unsqueeze(0) # Add batch dim (1, N, 3)
        # The decoder needs (z, points)
        # z is (1, C, R, R, R)
        sdf_out = vae.decoder(latent, chunk)
        sdf_values.append(sdf_out.squeeze(0).cpu().numpy())
        
    sdf_values = np.concatenate(sdf_values, axis=0)
    sdf_volume = sdf_values.reshape(resolution, resolution, resolution)
    
    print(f"SDF Stats: Min={sdf_volume.min():.4f}, Max={sdf_volume.max():.4f}, Mean={sdf_volume.mean():.4f}")

    if sdf_volume.min() > threshold or sdf_volume.max() < threshold:
        print("Warning: SDF volume does not cross the threshold. No surface found.")
        return None
        
    verts, faces, normals, values = measure.marching_cubes(sdf_volume, level=threshold)
    verts = verts / (resolution - 1) * 2 - 1
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=normals)
    return mesh

def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # 1. Load VAE
    print(f"Loading VAE from {args.vae_ckpt}...")
    try:
        vae = SdfVAE(latent_dim=256, num_points=16384, resolution=16)
        state_dict = torch.load(args.vae_ckpt, map_location='cpu')
        vae.load_state_dict(state_dict)
        vae.eval().to(device)
        print("VAE loaded.")
    except Exception as e:
        print(f"Error loading VAE: {e}")
        return

    # 2. Load Data Sample
    print(f"Loading data sample from {args.input_npz}...")
    try:
        data = np.load(args.input_npz)
        points = data['points']
        sdf = data['sdf']
        if len(sdf.shape) == 1:
            sdf = sdf[:, np.newaxis]
            
        # Subsample if needed
        if points.shape[0] > 16384:
            indices = np.random.choice(points.shape[0], 16384, replace=False)
            points = points[indices]
            sdf = sdf[indices]
            
        points_tensor = torch.from_numpy(points).float().unsqueeze(0).to(device) # (1, N, 3)
        sdf_tensor = torch.from_numpy(sdf).float().unsqueeze(0).to(device)       # (1, N, 1)
        print(f"Data loaded. Points: {points_tensor.shape}, SDF: {sdf_tensor.shape}")
    except Exception as e:
        print(f"Error loading data sample: {e}")
        return

    # 3. Encode
    print("Encoding...")
    mean, logvar = vae.encoder(points_tensor, sdf_tensor)
    
    # For reconstruction, we use the mean (deterministic)
    z = mean
    print(f"Latent shape: {z.shape}")

    # 4. Decode
    print("Decoding...")
    mesh = decode_to_mesh(vae, z, resolution=args.resolution, device=device)

    # 5. Save
    if mesh:
        os.makedirs(os.path.dirname(args.output_obj), exist_ok=True)
        mesh.export(args.output_obj)
        print(f"SUCCESS: Reconstructed mesh saved to {args.output_obj}")
    else:
        print("FAILURE: Could not reconstruct mesh.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Test VAE Reconstruction")
    parser.add_argument('--vae_ckpt', type=str, required=True, help='Path to VAE checkpoint')
    parser.add_argument('--input_npz', type=str, required=True, help='Path to input .npz file (with points and sdf)')
    parser.add_argument('--output_obj', type=str, default='reconstruction.obj', help='Output path for .obj file')
    parser.add_argument('--resolution', type=int, default=128, help='Grid resolution for marching cubes')
    args = parser.parse_args()
    
    main(args)
