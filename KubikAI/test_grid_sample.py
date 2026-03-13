import torch
import torch.nn.functional as F

B, C, R = 1, 1, 4
points = torch.tensor([[[-1.0, 0.0, 1.0]]])  # x=-1, y=0, z=1

# --- ENCODER LOGIC ---
coords = ((points + 1.0) / 2.0 * (R - 1)).clamp(0, R - 1).long()
print("Encoder Coords (x, y, z):", coords[0, 0].tolist())

flat_coords = coords[:, :, 0] * (R * R) + coords[:, :, 1] * R + coords[:, :, 2]
flat_grid = torch.zeros(B, C, R * R * R)

point_features = torch.tensor([[[100.0]]]).permute(0, 2, 1) # Feature value 100
flat_coords_expanded = flat_coords.unsqueeze(1).expand(-1, C, -1)
flat_grid.scatter_add_(2, flat_coords_expanded, point_features)

grid = flat_grid.view(B, C, R, R, R)

# Find where the feature went in the tensor (D, H, W)
nonzero = torch.nonzero(grid)
print("Tensor stored at index (D, H, W):", nonzero[0, 2:].tolist())

# --- DECODER LOGIC ---
grid_query = points.view(B, 1, 1, 1, 3) # (B, D_out, H_out, W_out, 3)

# Sample with raw points
sampled_raw = F.grid_sample(grid, grid_query, align_corners=True, padding_mode='zeros')
print("Sampled value with raw points:", sampled_raw[0, 0, 0, 0, 0].item())

# Sample with reversed points (z, y, x)
grid_query_reversed = points[..., [2, 1, 0]].view(B, 1, 1, 1, 3)
sampled_fixed = F.grid_sample(grid, grid_query_reversed, align_corners=True, padding_mode='zeros')
print("Sampled value with reversed points (W, H, D):", sampled_fixed[0, 0, 0, 0, 0].item())
