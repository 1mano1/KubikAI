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
    Encodes a set of SDF points and values into a 3D latent grid.
    Optimized with 3D Convolutions to save memory.
    """
    def __init__(self, latent_dim=256, point_dim=3, emb_dim=128, num_points=16384, resolution=16):
        super().__init__()
        self.num_points = num_points
        self.resolution = resolution
        self.latent_dim = latent_dim
        self.pos_emb = SinusoidalEmbedding(emb_dim)
        
        self.point_processor = MLP(in_dim=point_dim + 1 + (point_dim * emb_dim), out_dim=256, hidden_layers=2, hidden_dim=256)

        self.aggregator = nn.Sequential(
            nn.Linear(256, 512), 
            nn.GELU(),
            nn.Linear(512, 512),
            nn.GELU(),
        )

        # Map to a small 4x4x4 seed grid
        self.fc_seed = nn.Linear(512, latent_dim * 4 * 4 * 4)
        
        # Use Transposed Convolutions with Normalization
        self.upsampler = nn.Sequential(
            nn.ConvTranspose3d(latent_dim, latent_dim, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(8, latent_dim),
            nn.GELU(),
            nn.ConvTranspose3d(latent_dim, latent_dim, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(8, latent_dim),
            nn.GELU(),
        )
        
        # Final layers for mean and logvar on the 16x16x16 grid
        self.to_mean = nn.Conv3d(latent_dim, latent_dim, kernel_size=3, padding=1)
        self.to_logvar = nn.Conv3d(latent_dim, latent_dim, kernel_size=3, padding=1)

    def forward(self, points, sdf):
        B, N, _ = points.shape
        pos_embed = self.pos_emb(points).view(B, N, -1)
        point_sdf_cat = torch.cat([points, sdf, pos_embed], dim=-1)
        point_features = self.point_processor(point_sdf_cat)
        pooled_features, _ = torch.max(point_features, dim=1)
        aggregated = self.aggregator(pooled_features)

        # Generate the 16x16x16 grid using upsampling
        seed = self.fc_seed(aggregated).view(B, self.latent_dim, 4, 4, 4)
        grid = self.upsampler(seed) # (B, 256, 16, 16, 16)

        mean = self.to_mean(grid)
        logvar = self.to_logvar(grid)

        return mean, logvar

class SdfDecoder(nn.Module):
    """
    Decodes a latent grid and query points into SDF values.
    """
    def __init__(self, latent_dim=256, point_dim=3, emb_dim=256, resolution=16):
        super().__init__()
        self.resolution = resolution
        self.pos_emb = SinusoidalEmbedding(emb_dim)
        # We'll use 3D convolution layers with Norm and GELU for better latent extraction
        self.grid_processor = nn.Sequential(
            nn.Conv3d(latent_dim, latent_dim, kernel_size=3, padding=1),
            nn.GroupNorm(8, latent_dim),
            nn.GELU(),
            nn.Conv3d(latent_dim, latent_dim, kernel_size=3, padding=1),
            nn.GroupNorm(8, latent_dim),
            nn.GELU(),
        )
        # Removed Tanh activation to prevent vanishing gradients. Target SDFs should be clamped instead.
        self.net = MLP(
            in_dim=latent_dim + (point_dim * emb_dim),
            out_dim=1,
            hidden_layers=4,
            hidden_dim=512,
            final_activation=None
        )

    def forward(self, z, points):
        # z: (B, latent_dim, R, R, R), points: (B, N, 3)
        B, N, _ = points.shape
        
        # Process the latent grid
        z_processed = self.grid_processor(z) # (B, latent_dim, R, R, R)

        # Sample the latent grid at the query point positions
        grid_query = points.view(B, N, 1, 1, 3)
        z_sampled = F.grid_sample(z_processed, grid_query, align_corners=True)
        z_sampled = z_sampled.view(B, -1, N).permute(0, 2, 1) # (B, N, latent_dim)

        # Get positional embeddings for query points
        pos_embed = self.pos_emb(points)
        pos_embed = pos_embed.view(B, N, -1)

        # Concatenate latent features and positional embeddings
        z_pos_cat = torch.cat([z_sampled, pos_embed], dim=-1)

        # Predict SDF values
        sdf_pred = self.net(z_pos_cat)

        return sdf_pred

class SdfVAE(nn.Module):
    """
    A VAE for Signed Distance Functions.
    """
    def __init__(self, latent_dim=256, num_points=16384, resolution=16):
        super().__init__()
        self.encoder = SdfEncoder(latent_dim=latent_dim, num_points=num_points, resolution=resolution)
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

