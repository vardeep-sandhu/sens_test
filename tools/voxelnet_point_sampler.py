import numpy as np
import torch
import torch.nn as nn

import torch


def voxelization_fixed(
    points, voxel_size, grid_bounds, grid_size, max_points_per_voxel
):
    """
    Perform voxelization with a fixed output grid size.

    Args:
        points: (B, N, 3+C) tensor of input point cloud data.
        voxel_size: (dx, dy, dz) size of each voxel.
        grid_bounds: ((x_min, x_max), (y_min, y_max), (z_min, z_max)).
        grid_size: (x_bins, y_bins, z_bins) fixed grid size.
        max_points_per_voxel: Maximum number of points per voxel.

    Returns:
        voxel_features: (B, x_bins, y_bins, z_bins, max_points_per_voxel, C) padded voxel features.
        voxel_mask: (B, x_bins, y_bins, z_bins) mask for valid voxels.
    """
    batch_size, num_points, num_features = points.shape
    (x_min, x_max), (y_min, y_max), (z_min, z_max) = grid_bounds
    x_bins, y_bins, z_bins = grid_size

    # Quantize points into voxel indices
    voxel_indices = (
        (
            (
                points[:, :, :3]
                - torch.tensor([x_min, y_min, z_min], device=points.device)
            )
            / torch.tensor(voxel_size, device=points.device)
        )
        .floor()
        .long()
    )

    # Clip voxel indices to the grid size
    voxel_indices = torch.minimum(
        torch.maximum(voxel_indices, torch.tensor(0, device=points.device)),
        torch.tensor([x_bins - 1, y_bins - 1, z_bins - 1], device=points.device),
    )

    # Initialize voxel storage
    voxel_features = torch.zeros(
        (batch_size, x_bins, y_bins, z_bins, max_points_per_voxel, num_features - 3),
        device=points.device,
    )
    voxel_mask = torch.zeros((batch_size, x_bins, y_bins, z_bins), device=points.device)

    # Populate voxels
    for b in range(batch_size):
        for p in range(num_points):
            vx, vy, vz = voxel_indices[b, p]
            if voxel_mask[b, vx, vy, vz] < max_points_per_voxel:
                count = int(voxel_mask[b, vx, vy, vz])
                voxel_features[b, vx, vy, vz, count] = points[b, p, 3:]
                voxel_mask[b, vx, vy, vz] += 1

    # Convert mask to binary
    voxel_mask = (voxel_mask > 0).to(torch.float32)

    return voxel_features, voxel_mask


def voxel_grid_processing(voxel_features, max_voxels=200, max_points_per_voxel=32):
    # voxel_features shape: [V, P, 3]
    num_voxels, points_per_voxel, feature_dim = voxel_features.shape

    # Create a padded tensor with zeros
    padded_voxels = torch.zeros(
        max_voxels,
        max_points_per_voxel,
        feature_dim,
        dtype=voxel_features.dtype,
        device=voxel_features.device,
    )

    # Mask to track valid voxels
    valid_voxel_mask = torch.zeros(
        max_voxels, dtype=torch.bool, device=voxel_features.device
    )

    # Limit to max_voxels
    curr_voxels = min(num_voxels, max_voxels)
    valid_voxel_mask[:curr_voxels] = True

    for v in range(curr_voxels):
        # Limit to max_points_per_voxel
        curr_points = min(points_per_voxel[v], max_points_per_voxel)
        padded_voxels[v, :curr_points] = voxel_features[v, :curr_points]

    return padded_voxels, valid_voxel_mask


def voxelization(points, voxel_size, grid_bounds, max_points=32):
    """
    Perform voxelization of a point cloud using PyTorch.

    Args:
        points: (N, 3+C) tensor of raw point cloud data [x, y, z, features...].
        voxel_size: Size of each voxel (x, y, z).
        grid_bounds: ((x_min, x_max), (y_min, y_max), (z_min, z_max)).
        max_points: Maximum number of points per voxel.

    Returns:
        voxel_features: (M, max_points, C) tensor of features for each voxel.
        voxel_coords: (M, 3) tensor of voxel grid coordinates.
    """
    # Unpack grid bounds
    (x_min, x_max), (y_min, y_max), (z_min, z_max) = grid_bounds

    # Filter points within bounds
    valid_mask = (
        (points[:, 0] >= x_min)
        & (points[:, 0] < x_max)
        & (points[:, 1] >= y_min)
        & (points[:, 1] < y_max)
        & (points[:, 2] >= z_min)
        & (points[:, 2] < z_max)
    )
    points = points[valid_mask]

    # Compute voxel indices
    voxel_indices = torch.floor(
        (points[:, :3] - torch.tensor([x_min, y_min, z_min], device=points.device))
        / torch.tensor(voxel_size, device=points.device)
    ).long()

    # Group points by voxel index
    voxel_dict = {}
    for i, voxel_idx in enumerate(voxel_indices):
        voxel_key = tuple(voxel_idx.tolist())
        if voxel_key not in voxel_dict:
            voxel_dict[voxel_key] = []
        voxel_dict[voxel_key].append(points[i])

    # Generate voxel features and coordinates
    voxel_features = []
    voxel_coords = []

    for voxel_key, voxel_points in voxel_dict.items():
        voxel_coords.append(torch.tensor(voxel_key, device=points.device))
        voxel_points = torch.stack(voxel_points)

        # Limit points per voxel
        if voxel_points.size(0) > max_points:
            voxel_points = voxel_points[:max_points]
        elif voxel_points.size(0) < max_points:
            # Pad with zeros if not enough points
            padding = torch.zeros(
                (max_points - voxel_points.size(0), voxel_points.size(1)),
                device=points.device,
            )
            voxel_points = torch.cat((voxel_points, padding), dim=0)
        voxel_points = voxel_grid_processing(torch.Tensor(voxel_points))[0]
        voxel_features.append(voxel_points)

    voxel_features = torch.stack(
        voxel_grid_processing(voxel_features)
    )  # Shape: (M, max_points, C)
    voxel_coords = torch.stack(voxel_coords)  # Shape: (M, 3)

    return voxel_features, voxel_coords


class VoxelFeatureExtractor(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(VoxelFeatureExtractor, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
        )
        self.pooling = nn.MaxPool1d(32)  # Aggregate within each voxel

    def forward(self, voxel_features):
        """
        Args:
            voxel_features: (B, N, P, C) where
                B: Batch size
                N: Number of voxels
                P: Points per voxel
                C: Feature dimension
        Returns:
            voxel_out: (B, N, output_dim)
        """
        B, N, P, C = voxel_features.shape
        x = self.mlp(voxel_features.view(-1, C))  # Flatten to (B*N*P, C)
        x = x.view(B * N, P, -1).permute(0, 2, 1)  # Reshape for pooling
        x = self.pooling(x).squeeze(-1)  # Apply MaxPool1D
        return x.view(B, N, -1)  # Reshape to (B, N, output_dim)


class VoxelNetBackbone(nn.Module):
    def __init__(self, input_dim, voxel_dim):
        super(VoxelNetBackbone, self).__init__()
        self.vfe = VoxelFeatureExtractor(input_dim=input_dim, output_dim=voxel_dim)

        # 3D Convolutional layers
        self.conv3d = nn.Sequential(
            nn.Conv3d(voxel_dim, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.Conv3d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(),
        )

    def forward(self, voxel_features, voxel_coords):
        """
        Args:
            voxel_features: (B, N, P, C) as extracted point features
            voxel_coords: (B, N, 3) voxel grid coordinates
        Returns:
            conv_out: Processed voxel features (B, D, H, W)
        """
        # Step 1: Voxel Feature Extraction
        voxel_features = self.vfe(voxel_features)  # (B, N, voxel_dim)

        # Step 2: 3D Convolutions
        # Convert voxel features to a dense 4D tensor
        B, N, C = voxel_features.shape
        D, H, W = 50, 50, 50  # Adjust as per grid size
        voxel_grid = torch.zeros((B, C, D, H, W), device=voxel_features.device)
        for b in range(B):
            for n in range(N):
                x, y, z = voxel_coords[b, n]  # voxel grid coordinates
                voxel_grid[b, :, x, y, z] = voxel_features[b, n]

        conv_out = self.conv3d(voxel_grid)  # Apply 3D convolutions
        return conv_out
