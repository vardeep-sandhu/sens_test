import torch
import torch.nn as nn

class VoxelFeatureExtractor(nn.Module):
    def __init__(self, input_dim=3, output_dim=64):
        """
        Args:
            input_dim (int): The dimensionality of the input features (x, y, z).
            output_dim (int): The dimensionality of the output features.
        """
        super(VoxelFeatureExtractor, self).__init__()
        # Define an MLP to learn transformations for point features
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim),
            nn.ReLU()
        )

    def forward(self, voxels, num_voxels, voxel_mask):
        """
        Args:
            voxels (torch.Tensor): Input tensor of shape (B, N, P, F).
                                   B: Batch size, N: Number of voxels, P: Max points per voxel, F: Feature dimension.
            num_voxels (torch.Tensor): Tensor of shape (B, N), indicating the number of real points in each voxel.
            voxel_mask (torch.Tensor): Tensor of shape (B, N), indicating valid voxels in the batch.

        Returns:
            torch.Tensor: Extracted features of shape (B, N, output_dim).
        """
        B, N, P, F = voxels.shape  # Batch size, Number of voxels, Max points per voxel, Feature dimension

        # Mask to identify valid points within each voxel
        point_mask = torch.arange(P, device=num_voxels.device).expand(B, N, P) < num_voxels.unsqueeze(-1)

        # Apply the point mask to set invalid points to zero
        # valid_points = voxels * point_mask.unsqueeze(-1)  # Shape (B, N, P, F) 

        # Compute per-point features using the MLP
        point_features = self.mlp(voxels)  # Shape (B, N, P, output_dim)

        # Mask invalid points to exclude them from the aggregation
        point_features = point_features * point_mask.unsqueeze(-1)  # Shape (B, N, P, output_dim)

        # Aggregate features within each voxel (e.g., mean pooling)
        aggregated_features = point_features.sum(dim=2) / num_voxels.unsqueeze(-1).clamp(min=1)  # Shape (B, N, output_dim)

        # Mask out invalid voxels in the final output
        voxel_features = aggregated_features * voxel_mask.unsqueeze(-1)  # Shape (B, N, output_dim)

        return voxel_features

class TrajectoryPredictionModel(nn.Module):
    def __init__(self, input_channels, num_classes=30 * 7):
        """
        Processes a smaller BEV grid and predicts trajectory.
        
        Args:
            input_channels (int): Number of input channels (e.g., 3 for the BEV grid).
            num_classes (int): Output dimension (e.g., 30 * 7 for trajectory).
        """
        super(TrajectoryPredictionModel, self).__init__()
        
        # Adjusted 2D Convolutional Backbone for smaller grid
        self.backbone = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, stride=2, padding=1),  # -> B, 32, H/2, W/2
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # -> B, 64, H/4, W/4
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # -> B, 128, H/8, W/8
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),  # -> B, 256, H/16, W/16
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),  # -> B, 512, H/32, W/32
            nn.ReLU(),
        )
        
        # Compute feature map size after Conv layers (assuming square input grid)
        # Example: If input grid size is 200x200, final size -> (H/32, W/32) = (6, 6)
        self.feature_map_size = 16  # Replace with the actual feature map size after downsampling
        
        # Fully Connected Layers for Trajectory Prediction
        self.trajectory_head = nn.Sequential(
            nn.Flatten(),  # Flatten -> B, 512 * feature_map_size * feature_map_size
            nn.Linear(512 * self.feature_map_size * self.feature_map_size, 1024),
            nn.ReLU(),
            nn.Linear(1024, num_classes),  # Output -> B, 30 * 7
        )
    
    def forward(self, x):
        """
        Forward pass for trajectory prediction.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W).
        
        Returns:
            torch.Tensor: Output tensor of shape (B, 30 * 7).
        """
        x = self.backbone(x)  # Pass through 2D ConvNet
        x = self.trajectory_head(x)  # Flatten and predict trajectory
        return x


class BEVConverter(nn.Module):
    def __init__(self, input_channels, bev_method="max"):
        """
        Converts a 3D voxel grid to a 2D Bird's Eye View (BEV) representation.
        
        Args:
            input_channels (int): Number of input channels (features per voxel).
            bev_method (str): Method to convert to BEV. Options: "max", "sum", "learned".
        """
        super(BEVConverter, self).__init__()
        self.bev_method = bev_method
        
        if self.bev_method == "learned":
            # Learnable weights for depth axis aggregation
            self.depth_conv = nn.Conv3d(input_channels, input_channels, kernel_size=(40, 1, 1), stride=1, padding=0)
        
    def forward(self, x):
        """
        Forward pass for BEV conversion.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, C, D, H, W).
        
        Returns:
            torch.Tensor: BEV tensor of shape (B, C, H, W).
        """
        if self.bev_method == "max":
            # Max pooling along the depth dimension (dim=2)
            bev = torch.max(x, dim=2).values
        elif self.bev_method == "sum":
            # Summation along the depth dimension
            bev = torch.sum(x, dim=2)
        elif self.bev_method == "learned":
            # Learnable depth aggregation using Conv3D
            bev = self.depth_conv(x).squeeze(2)
        else:
            raise ValueError(f"Unknown BEV method: {self.bev_method}")
        
        return bev
