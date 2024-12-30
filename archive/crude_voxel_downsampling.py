import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F
from loss import *


class MLP(nn.Module):
    def __init__(self, input_feature_len, out_feature_len=128):
        super().__init__()
        self.out_feature_len = out_feature_len
        self.fc1 = nn.Linear(input_feature_len, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, out_feature_len)

    def forward(self, x):
        features = F.relu(self.fc1(x))
        features = F.relu(self.fc2(features))
        features = self.fc3(features)
        return x, features


class OptimizedTrajectoryPredictor(pl.LightningModule):
    def __init__(
        self,
        input_dim=3,
        hidden_dim=128,  # Reduced hidden dimension
        # output_dim=30 * 7,
        num_layers=2,  # Reduced layers
        num_heads=4,  # Reduced heads
        max_points=4096,
        lookahead=30,
        outdim=7,
    ):
        super().__init__()
        self.lookahead = lookahead
        self.outdim = outdim
        # Point Cloud Sampler
        self.feature_encoder = MLP(3, 128)

        # Point Cloud Feature Encoding
        self.point_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.SiLU(), nn.LayerNorm(hidden_dim)
        )

        # Use Lightweight Attention
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads, dropout=0.1, batch_first=True
        )

        # Trajectory Prediction Head
        self.trajectory_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, lookahead * outdim),
        )

        # Loss Criterion
        self.criterion = trajectory_loss

        # Mixed Precision Training
        self.automatic_optimization = False

    def voxel_grid_filter_batched(self, point_cloud, voxel_size, max_points=None):
        """
        Downsample a batched point cloud using voxel grid filtering and pad outputs for batchability.

        Parameters:
            point_cloud (torch.Tensor): Tensor of shape (B, N, 3), where B is the batch size and N is the number of points.
            voxel_size (float): Size of each voxel.
            max_points (int): Maximum number of points to pad to. If None, it uses the max number of points in the batch.

        Returns:
            torch.Tensor: Downsampled point cloud of shape (B, P, 3), where P is max_points.
            torch.Tensor: Mask of shape (B, P), indicating valid points (1 for valid, 0 for padded).
        """
        batch_size, _, _ = point_cloud.shape

        # Store results for each batch
        downsampled_batches = []
        batch_sizes = []

        for b in range(batch_size):
            # Extract the current batch's point cloud
            single_cloud = point_cloud[b]

            # Quantize the points to voxel grid indices
            voxel_indices = torch.floor(single_cloud / voxel_size).int()

            # Create a dictionary to store unique voxels
            voxel_dict = {}
            for i, voxel in enumerate(voxel_indices):
                voxel_tuple = tuple(
                    voxel.tolist()
                )  # Convert to a tuple, which is hashable
                if voxel_tuple not in voxel_dict:
                    voxel_dict[voxel_tuple] = (
                        i  # Store index of the first point in this voxel
                    )

            # Retrieve unique points
            unique_indices = list(voxel_dict.values())
            downsampled_point_cloud = single_cloud[unique_indices]

            downsampled_batches.append(downsampled_point_cloud)
            batch_sizes.append(len(unique_indices))

        # Determine padding size
        if max_points is None:
            max_points = max(batch_sizes)

        # Initialize padded output and mask
        padded_batch = torch.zeros(
            (batch_size, max_points, 3),
            dtype=point_cloud.dtype,
            device=point_cloud.device,
        )
        mask = torch.zeros(
            (batch_size, max_points), dtype=torch.bool, device=point_cloud.device
        )

        for b, downsampled in enumerate(downsampled_batches):
            length = downsampled.shape[0]
            padded_batch[b, :length, :] = downsampled
            mask[b, :length] = 1

        return padded_batch, mask

    def forward(self, x):
        """
        Forward pass for the model.

        Parameters:
            x (torch.Tensor): Input point cloud tensor of shape (B, N, input_dim).

        Returns:
            torch.Tensor: Predicted trajectory of shape (B, output_dim).
        """
        # Sample points to reduce memory
        x, mask = self.voxel_grid_filter_batched(
            x, 1.0
        )  # Downsample and get mask (B, P, 3), (B, P)

        # Encode point features (accounting for padded points)
        features = self.point_encoder(x)  # Shape: (B, P, hidden_dim)
        features = features * mask.unsqueeze(-1)  # Zero out padded points in features

        # Attention with memory efficiency
        attn_output, _ = self.attention(
            features, features, features, key_padding_mask=~mask
        )
        # Shape: (B, P, hidden_dim)

        # Global pooling (masked mean pooling)
        valid_counts = mask.sum(dim=1, keepdim=True)  # Shape: (B, 1)
        global_features = (
            attn_output.sum(dim=1) / valid_counts
        )  # Shape: (B, hidden_dim)

        # Predict trajectory
        trajectory = self.trajectory_head(global_features)  # Shape: (B, output_dim)

        return trajectory

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
