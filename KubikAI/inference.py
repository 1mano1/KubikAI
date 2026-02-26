import os
import sys
import json
import argparse
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F
from skimage import measure
import trimesh

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from KubikAI.models.sdf_vae import SdfVAE
from KubikAI.models.cross_attention_flow import CrossAttentionFlowModel
from easydict import EasyDict as edict

class KubikInference:
    def __init__(self, config_path, vae_ckpt, flow_ckpt, device='cuda'):
        self.device = device
        self.config = self._load_config(config_path)
        
        print("Loading models...")
        self.vae = self._load_vae(vae_ckpt)
        self.flow = self._load_flow(flow_ckpt)
        self.image_encoder = self._load_image_encoder()
        print("Models loaded successfully.")

    def _load_config(self, path):
        with open(path, 'r') as f:
            cfg = json.load(f)
        return edict(cfg)

    def _load_vae(self, ckpt_path):
        # Initialize VAE with same params as training
        # Assuming standard config here, could make configurable if needed
        model = SdfVAE(latent_dim=256, num_points=16384, resolution=16)
        state_dict = torch.load(ckpt_path, map_location='cpu')
        model.load_state_dict(state_dict)
        model.eval().to(self.device)
        return model

    def _load_flow(self, ckpt_path):
        flow_cfg = list(self.config.models.values())[0]
        model = CrossAttentionFlowModel(**flow_cfg.args)
        
        # Load flow checkpoint
        # The trainer saves checkpoints as a dictionary containing 'models' key if using BaseTrainer
        # Or directly if saving state dict. Usually BaseTrainer saves full checkpoint.
        checkpoint = torch.load(ckpt_path, map_location='cpu')
        
        if 'models' in checkpoint:
            state_dict = checkpoint['models']['flow']
        elif 'state_dict' in checkpoint: # Common alternative
             state_dict = checkpoint['state_dict']
             # Remove 'flow.' prefix if present
             state_dict = {k.replace('flow.', ''): v for k, v in state_dict.items()}
        else:
            # Assume it's the state dict itself
            state_dict = checkpoint
            
        model.load_state_dict(state_dict)
        model.eval().to(self.device)
        return model

    def _load_image_encoder(self):
        model_name = self.config.trainer.args.image_cond_model
        print(f"Loading image encoder: {model_name}...")
        model = torch.hub.load('facebookresearch/dinov2', model_name, pretrained=True)
        model.eval().to(self.device)
        
        transform = transforms.Compose([
            transforms.Resize((224, 224)), # Ensure correct input size for DINOv2
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        return {'model': model, 'transform': transform}

    @torch.no_grad()
    def encode_image(self, image_path):
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.image_encoder['transform'](image).unsqueeze(0).to(self.device)
        
        # Extract features exactly as in training
        # Note: is_training=True returns the dict with 'x_prenorm'
        features = self.image_encoder['model'](image_tensor, is_training=True)['x_prenorm']
        
        # Normalize features
        features = F.layer_norm(features, features.shape[-1:])
        return features

    @torch.no_grad()
    def generate_latent(self, cond_features, steps=50):
        """
        Generate latent representation using Euler integration of the flow model.
        """
        batch_size = cond_features.shape[0]
        latent_shape = (batch_size, 256, 16, 16, 16)
        
        # Initial noise x_1 (at t=1)
        x = torch.randn(latent_shape, device=self.device)
        
        # Time steps from 1 to 0
        times = torch.linspace(1, 0, steps + 1, device=self.device)
        dt = 1.0 / steps

        # Prepare conditions
        global_cond = cond_features[:, 0]       # CLS token
        local_cond = cond_features[:, 1:]       # Patch tokens

        for i in range(steps):
            t = times[i]
            t_batch = torch.ones(batch_size, device=self.device) * t
            
            # Predict velocity v_t
            # Note: The model expects t in [0, 1000] range usually if using sinusoidal embeddings tailored for diffusion,
            # but our flow model uses continuous t. Let's check training logic.
            # In training: pred = flow_model(x_t, t * 1000, ...)
            # So we multiply by 1000.
            v_pred = self.flow(x, t_batch * 1000, global_cond, local_cond)
            
            # Euler step: x_{t-dt} = x_t - v_pred * dt
            # Since we go from 1 to 0, dt is effectively negative in time, so we subtract v*dt where dt is positive step size?
            # flow definition: dx/dt = v.
            # Backward Euler: x(t - dt) = x(t) - v(x(t), t) * dt
            x = x - v_pred * dt
            
        return x

    @torch.no_grad()
    def decode_to_mesh(self, latent, resolution=64, threshold=0.0):
        """
        Decode the latent grid to a mesh using Marching Cubes.
        """
        # Create a grid of query points
        # range [-1, 1] matches the VAE's expected input range
        grid_points = np.linspace(-1, 1, resolution)
        grid_points = np.stack(np.meshgrid(grid_points, grid_points, grid_points, indexing='ij'), axis=-1)
        grid_points = grid_points.reshape(-1, 3)
        
        # Process in chunks to avoid OOM
        chunk_size = 100000
        sdf_values = []
        
        points_tensor = torch.from_numpy(grid_points).float().to(self.device)
        
        # The decoder expects (B, N, 3)
        # latent is (B, C, R, R, R)
        
        for i in range(0, points_tensor.shape[0], chunk_size):
            chunk = points_tensor[i:i+chunk_size].unsqueeze(0) # Add batch dim
            sdf_out = self.vae.decoder(latent, chunk)
            sdf_values.append(sdf_out.squeeze(0).cpu().numpy())
            
        sdf_values = np.concatenate(sdf_values, axis=0)
        sdf_volume = sdf_values.reshape(resolution, resolution, resolution)
        
        # Extract mesh
        # If all values are positive or all negative, marching cubes will fail
        if sdf_volume.min() > threshold or sdf_volume.max() < threshold:
            print("Warning: SDF volume does not cross the threshold. No surface found.")
            return None
            
        verts, faces, normals, values = measure.marching_cubes(sdf_volume, level=threshold)
        
        # Normalize verts back to [-1, 1]
        # verts are in grid coordinates [0, resolution-1]
        verts = verts / (resolution - 1) # [0, 1]
        verts = verts * 2 - 1 # [-1, 1]
        
        mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=normals)
        return mesh

    def run(self, image_path, output_path):
        print(f"Processing {image_path}...")
        features = self.encode_image(image_path)
        
        print("Generating latent representation...")
        latent = self.generate_latent(features)
        
        print("Reconstructing mesh...")
        mesh = self.decode_to_mesh(latent)
        
        if mesh:
            mesh.export(output_path)
            print(f"Saved mesh to {output_path}")
        else:
            print("Failed to generate mesh.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="KubikAI Inference Script")
    parser.add_argument('--image', type=str, required=True, help='Path to input image')
    parser.add_argument('--output', type=str, required=True, help='Path to output .obj file')
    parser.add_argument('--config', type=str, default='KubikAI/configs/kubikai_flow_v1.json', help='Path to flow config')
    parser.add_argument('--vae_ckpt', type=str, default='outputs/sdf_vae_training/ckpts/vae_step0050000.pt', help='Path to VAE checkpoint')
    parser.add_argument('--flow_ckpt', type=str, required=True, help='Path to Flow checkpoint')
    
    args = parser.parse_args()
    
    inference = KubikInference(args.config, args.vae_ckpt, args.flow_ckpt)
    inference.run(args.image, args.output)
