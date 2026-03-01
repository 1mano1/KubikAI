import torch
import torch.nn as nn
import torch.nn.functional as F

# Helper module for sinusoidal positional encoding
class SinusoidalEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = torch.log(torch.tensor(10000.0, device=device)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[..., None] * emb[None, ...]
        return torch.cat((emb.sin(), emb.cos()), dim=-1)

# Basic MLP class
class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_layers, hidden_dim, final_activation=None):
        super().__init__()
        layers = [nn.Linear(in_dim, hidden_dim), nn.GELU()]
        for _ in range(hidden_layers):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.GELU()])
        layers.append(nn.Linear(hidden_dim, out_dim))
        
        # Add the final activation if specified
        if final_activation:
            layers.append(final_activation())
            
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class SdfEncoder(nn.Module):
    """
    Encodes a set of SDF points and values into a 3D latent grid using Point-Voxel Splatting.
    This preserves spatial relationships natively, preventing the 'muddy' effect and capturing 
    high-frequency details like fingers and faces.
    """
    def __init__(self, latent_dim=128, point_dim=3, emb_dim=32, resolution=32):
        super().__init__()
        self.resolution = resolution
        self.latent_dim = latent_dim
        
        # Initial point feature extraction
        self.point_mlp = nn.Sequential(
            nn.Linear(point_dim + 1, 64),
            nn.GELU(),
            nn.Linear(64, latent_dim)
        )
        
        # 3D CNN to process the voxelized features natively in 3D
        self.cnn = nn.Sequential(
            nn.Conv3d(latent_dim, latent_dim * 2, kernel_size=3, padding=1),
            nn.GroupNorm(8, latent_dim * 2),
            nn.GELU(),
            nn.Conv3d(latent_dim * 2, latent_dim * 2, kernel_size=3, padding=1),
            nn.GroupNorm(8, latent_dim * 2),
            nn.GELU(),
        )
        
        # Final layers for mean and logvar on the 32x32x32 grid
        self.to_mean = nn.Conv3d(latent_dim * 2, latent_dim, kernel_size=3, padding=1)
        self.to_logvar = nn.Conv3d(latent_dim * 2, latent_dim, kernel_size=3, padding=1)

    def forward(self, points, sdf):
        B, N, _ = points.shape
        device = points.device
        
        # 1. Point features
        x = torch.cat([points, sdf], dim=-1) # (B, N, 4)
        point_features = self.point_mlp(x) # (B, N, latent_dim)
        
        # 2. Voxelization (Splatting)
        # Map points from [-1, 1] to [0, resolution-1]
        coords = ((points + 1.0) / 2.0 * (self.resolution - 1)).clamp(0, self.resolution - 1).long()
        
        C = self.latent_dim
        R = self.resolution
        
        # Flattened spatial dim: R*R*R
        flat_coords = coords[:, :, 0] * (R * R) + coords[:, :, 1] * R + coords[:, :, 2] # (B, N)
        flat_grid = torch.zeros(B, C, R * R * R, device=device)
        
        # Permute point_features to (B, C, N) for scatter_add_
        point_features = point_features.permute(0, 2, 1)
        
        # Expand flat_coords for channels: (B, C, N)
        flat_coords_expanded = flat_coords.unsqueeze(1).expand(-1, C, -1)
        
        # Scatter add features into the grid
        flat_grid.scatter_add_(2, flat_coords_expanded, point_features)
        
        # Reshape back to 3D grid
        grid = flat_grid.view(B, C, R, R, R)
        
        # 3. Process with 3D CNN to diffuse and refine features
        features = self.cnn(grid)
        
        mean = self.to_mean(features)
        logvar = self.to_logvar(features)

        return mean, logvar

class SdfDecoder(nn.Module):
    """
    Decodes a latent grid and query points into SDF values using skip-connections via grid sampling.
    """
    def __init__(self, latent_dim=128, point_dim=3, emb_dim=32, resolution=32):
        super().__init__()
        self.resolution = resolution
        self.pos_emb = SinusoidalEmbedding(emb_dim)
        
        # 3D convolution layers with Norm and GELU to refine the sampled latent grid
        self.grid_processor = nn.Sequential(
            nn.Conv3d(latent_dim, latent_dim, kernel_size=3, padding=1),
            nn.GroupNorm(8, latent_dim),
            nn.GELU(),
            nn.Conv3d(latent_dim, latent_dim, kernel_size=3, padding=1),
            nn.GroupNorm(8, latent_dim),
            nn.GELU(),
        )
        
        # Decoder MLP
        self.net = MLP(
            in_dim=latent_dim + (point_dim * emb_dim),
            out_dim=1,
            hidden_layers=4,
            hidden_dim=256,
            final_activation=None
        )

    def forward(self, z, points):
        B, N, _ = points.shape
        
        # Process the latent grid
        z_processed = self.grid_processor(z) # (B, latent_dim, R, R, R)

        # Sample the latent grid at the query point positions
        # grid_sample in 3D expects (W, H, D) order for coordinates.
        # Since our tensor is built as (D, H, W) where x->D, y->H, z->W, 
        # we must reverse the last dimension of points: (z, y, x).
        grid_query = points[..., [2, 1, 0]].view(B, N, 1, 1, 3)
        
        # z_sampled: (B, C, N, 1, 1) -> (B, N, C)
        z_sampled = F.grid_sample(z_processed, grid_query, align_corners=True, padding_mode='zeros')
        z_sampled = z_sampled.view(B, -1, N).permute(0, 2, 1) 

        # Get positional embeddings for query points
        pos_embed = self.pos_emb(points)
        pos_embed = pos_embed.view(B, N, -1)

        # Concatenate local features from the grid and precise positional embeddings
        z_pos_cat = torch.cat([z_sampled, pos_embed], dim=-1)

        # Predict SDF values
        sdf_pred = self.net(z_pos_cat)

        return sdf_pred

class SdfVAE(nn.Module):
    """
    A Convolutional VAE for Signed Distance Functions.
    Designed to capture fine local details (hands, feet, faces) by preserving spatial geometry.
    """
    def __init__(self, latent_dim=128, num_points=16384, resolution=32):
        super().__init__()
        self.encoder = SdfEncoder(latent_dim=latent_dim, resolution=resolution)
        self.decoder = SdfDecoder(latent_dim=latent_dim, resolution=resolution)

    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std

    def forward(self, points, sdf):
        mean, logvar = self.encoder(points, sdf)
        z = self.reparameterize(mean, logvar)
        sdf_pred = self.decoder(z, points)
        return sdf_pred, mean, logvar
