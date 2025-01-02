import torch
import torch.nn as nn

class VoxelFeatureExtractor(nn.Module):
    def __init__(self, input_dim=3, output_dim=64):
        super(VoxelFeatureExtractor, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim),
            nn.ReLU()
        )
        self.attention_layer = nn.Sequential(
            nn.Linear(output_dim, output_dim),  # Project to attention space
            nn.Tanh(),                         # Nonlinearity for richer representations
            nn.Linear(output_dim, 1),          # Scalar attention score per point
            nn.Softmax(dim=2)                  # Normalize across points
        )

    def forward(self, voxels, num_voxels, voxel_mask):
        B, N, P, F = voxels.shape
        point_mask = torch.arange(P, device=num_voxels.device).expand(B, N, P) < num_voxels.unsqueeze(-1)
        point_features = self.mlp(voxels) * point_mask.unsqueeze(-1)

        # Attention mechanism
        attention_scores = self.attention_layer(point_features)  # Shape: (B, N, P, 1)
        point_features = point_features * attention_scores  # Apply attention weights

        # Aggregate features
        aggregated_features = point_features.sum(dim=2) / num_voxels.unsqueeze(-1).clamp(min=1)
        voxel_features = aggregated_features * voxel_mask.unsqueeze(-1)

        return voxel_features

class TrajectoryPredictionModel(nn.Module):
    def __init__(self, input_channels, num_classes=30 * 7):
        super(TrajectoryPredictionModel, self).__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )
        self.self_attention = nn.MultiheadAttention(embed_dim=512, num_heads=8, batch_first=True)
        self.flatten = nn.Flatten()
        self.trajectory_head = nn.Sequential(
            nn.Linear(512 * 16 * 16, 1024),
            nn.ReLU(),
            nn.Linear(1024, num_classes),
        )

    def forward(self, x):
        x = self.backbone(x)  # B, 512, H/32, W/32
        B, C, H, W = x.shape
        x = x.view(B, C, -1).permute(0, 2, 1)  # Reshape to (B, HW, C) for attention
        x, _ = self.self_attention(x, x, x)  # Apply self-attention
        x = x.permute(0, 2, 1).reshape(B, C, H, W)  # Reshape back to (B, C, H, W)
        x = self.flatten(x)
        x = self.trajectory_head(x)
        return x

class BEVConverter(nn.Module):
    def __init__(self, input_channels, bev_method="max"):
        super(BEVConverter, self).__init__()
        self.bev_method = bev_method
        if self.bev_method == "learned":
            self.depth_conv = nn.Conv3d(input_channels, input_channels, kernel_size=(40, 1, 1), stride=1, padding=0)
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool3d((1, None, None)),
            nn.Conv3d(input_channels, input_channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        if self.bev_method == "max":
            bev = torch.max(x, dim=2).values
        elif self.bev_method == "sum":
            bev = torch.sum(x, dim=2)
        elif self.bev_method == "learned":
            bev = self.depth_conv(x).squeeze(2)
        else:
            raise ValueError(f"Unknown BEV method: {self.bev_method}")
        
        # Channel-wise attention
        attention_weights = self.channel_attention(x)
        bev = bev * attention_weights.squeeze(2)

        return bev
