import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F
from voxelnet_point_sampler import voxelization, voxel_grid_processing
from loss import *


class VoxelFeatureEncoder(nn.Module):
    def __init__(self, input_dim, voxel_feature_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, voxel_feature_dim // 2),
            nn.ReLU(),
            nn.Linear(voxel_feature_dim // 2, voxel_feature_dim),
        )

    def forward(self, voxel_features):
        # Input: (B, V, P, input_dim) -> Output: (B, V, voxel_feature_dim)
        B, V, P, D = voxel_features.shape
        voxel_features = voxel_features.view(-1, P, D)  # Flatten batch and voxels
        encoded_features = self.mlp(voxel_features)  # (B * V, P, voxel_feature_dim)
        return encoded_features.max(dim=1)[0].view(B, V, -1)  # Max pooling


class VoxelNetBackbone(nn.Module):
    def __init__(self, voxel_feature_dim):
        super().__init__()
        self.conv3d = nn.Sequential(
            nn.Conv3d(voxel_feature_dim, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv3d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv3d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )

    def forward(self, voxel_grid):
        # Input: (B, voxel_feature_dim, D, H, W) -> Output: (B, C, D', H', W')
        return self.conv3d(voxel_grid)


class TrajectoryPredictorWithVoxelNet(pl.LightningModule):
    def __init__(
        self,
        voxel_size,
        grid_bounds,
        voxel_feature_dim=128,
        hidden_dim=256,
        lookahead=30,
        outdim=7,
    ):
        super().__init__()
        self.voxel_size = voxel_size
        self.grid_bounds = grid_bounds
        self.lookahead = lookahead
        self.outdim = outdim

        # Voxel Feature Encoder
        self.feature_encoder = VoxelFeatureEncoder(
            input_dim=3, voxel_feature_dim=voxel_feature_dim
        )

        # VoxelNet Backbone
        self.backbone = VoxelNetBackbone(voxel_feature_dim=voxel_feature_dim)

        # Trajectory Prediction Head
        self.trajectory_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, lookahead * outdim),
        )

        # Loss Criterion
        self.criterion = trajectory_loss

    def voxelization(self, point_cloud):
        """
        Perform voxelization and return voxelized features and voxel grid.
        """
        batch_size, _, _ = point_cloud.shape
        voxel_features, voxel_coords = [], []

        for b in range(batch_size):
            points = point_cloud[b]
            # Voxelization logic here (use previous implementation)
            voxelized_points, _ = voxelization(
                points, self.voxel_size, self.grid_bounds
            )
            voxel_features.append(voxelized_points)

        voxel_features = torch.stack(voxel_features)  # (B, V, P, 3)
        return voxel_features

    def forward(self, point_cloud):
        # Step 1: Voxelization
        voxel_features = self.voxelization(point_cloud)  # (B, V, P, 3)
        print(voxel_features.shape)
        padded_voxels, mask = voxel_grid_processing(
            voxel_features, max_voxels=1000, max_points_per_voxel=32
        )
        print(padded_voxels.shape)
        # Step 2: Encode Voxel Features
        encoded_features = self.feature_encoder(
            voxel_features
        )  # (B, V, voxel_feature_dim)

        # Step 3: Convert to Voxel Grid
        # Example: Convert voxel features to 3D grid (e.g., D x H x W)
        print("Input size:", encoded_features.shape)
        print("Input elements:", encoded_features.numel())
        print(
            "Expected elements for [B, C, 16, 16, 16]:",
            encoded_features.shape[0] * 16 * 16 * 16,
        )
        voxel_grid = encoded_features.view(encoded_features.shape[0], -1, 16, 16, 16)

        # Step 4: Process Voxel Grid through Backbone
        voxel_features = self.backbone(voxel_grid)  # (B, C, D', H', W')

        # Step 5: Global Pooling for Trajectory Prediction
        pooled_features = voxel_features.view(voxel_features.shape[0], -1)  # Flatten
        trajectory = self.trajectory_head(pooled_features)  # (B, lookahead * outdim)

        return trajectory.view(-1, self.lookahead, self.outdim)

    def training_step(self, batch, batch_idx):
        # Unpack batch
        lidar_points, mask, target = batch
        # Manual mixed precision
        optimizer = self.optimizers()
        optimizer.zero_grad()

        with torch.cuda.amp.autocast():
            prediction = self(lidar_points)
            prediction = prediction.view(-1, self.lookahead, self.outdim)
            loss = self.criterion(prediction, target)

        self.manual_backward(loss)
        optimizer.step()

        self.log("train_loss", loss, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):

        lidar_points, mask, target = batch

        prediction = self(lidar_points)
        prediction = prediction.view(-1, self.lookahead, self.outdim)
        ade, fde = compute_ade_fde(prediction, target)

        loss = self.criterion(prediction, target)

        self.log("val_loss", loss, on_epoch=True, on_step=False)
        self.log("val_ade", ade.mean(), on_epoch=True, on_step=False)
        self.log("val_fde", fde.mean(), on_epoch=True, on_step=False)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=5e-4,  # Slightly reduced learning rate
            weight_decay=1e-5,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=100, eta_min=1e-5
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
