import torch
import pytorch_lightning as pl
from loss import *
from voxelnet_point_sampler import VoxelFeatureExtractor,BEVConverter, TrajectoryPredictionModel

class TrajectoryPredictorWithVoxelNet(pl.LightningModule):
    def __init__(
        self,
        hidden_dim=256,
        lookahead=30,
        outdim=7,
    ):
        super().__init__()
        self.lookahead = lookahead
        self.outdim = outdim
        self.reader = VoxelFeatureExtractor()

        self.bev_converter = BEVConverter(64, "max")
        self.trajectory_head = TrajectoryPredictionModel(64)
        self.criterion = trajectory_loss
        self.device_ = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def extract_feat(self, data):
        input_features = self.reader(data["voxels"], data['num_points'], data["mask"])
        return input_features #B, N, 3 (3 is the feature dim)
    
    def grid_action(self, data, voxel_features):
        """
        Efficiently scatter voxel features into a dense grid.
        
        Args:
            data (dict): A dictionary containing:
                - 'grid_shape': Tuple of (H, W, D) defining the grid dimensions.
                - 'mask': Tensor of shape (B, N), indicating valid voxels.
                - 'voxel_coordinates': Tensor of shape (B, N, 3), containing voxel indices (z, y, x).
            voxel_features (torch.Tensor): Tensor of shape (B, N, output_dim), containing features for each voxel.

        Returns:
            torch.Tensor: Dense grid of shape (B, output_dim, D, W, H).
        """
        B, N, output_dim = voxel_features.shape
        H, W, D = map(int, data['grid_shape'])

        # Flatten 3D voxel coordinates to a 1D grid index
        voxel_coordinates = data['voxel_coordinates'].long()  # Shape: (B, N, 3)
        mask = data['mask'].bool()  # Shape: (B, N)

        # Compute flat indices for all valid voxels
        flat_indices = (
            voxel_coordinates[:, :, 0] * (W * H) +
            voxel_coordinates[:, :, 1] * H +
            voxel_coordinates[:, :, 2]
        )  # Shape: (B, N)

        # Apply mask to filter valid features and indices
        flat_indices = flat_indices[mask]  # Shape: (num_valid_voxels,)
        valid_features = voxel_features[mask]  # Shape: (num_valid_voxels, output_dim)

        # Batch indices for scatter operation
        batch_indices = torch.arange(B, device=voxel_features.device).repeat_interleave(mask.sum(dim=1))  # Shape: (num_valid_voxels,)

        # Create a dense grid and scatter features
        grid = torch.zeros((B, output_dim, D, W, H), device=voxel_features.device)
        grid_flat = grid.view(B, output_dim, -1)  # Shape: (B, output_dim, D * W * H)

        # Scatter valid features into the flattened grid
        grid_flat[batch_indices, :, flat_indices] = valid_features

        return grid

    def forward(self, data):
        encoded_features = self.extract_feat(
            data
        ) # B, N, 64

        grid = self.grid_action(data, encoded_features)
        grid = self.bev_converter(grid)
        out = self.trajectory_head(grid)
        return out

    def training_step(self, batch, batch_idx):
        batch = {key: value.to(self.device_) for key, value in batch.items()}

        prediction = self(batch)
        prediction = prediction.view(-1, self.lookahead, self.outdim)
        loss = self.criterion(prediction, batch["target"])

        self.log("train_loss", loss, prog_bar=True)
        return loss
    def validation_step(self, batch):
        batch = {key: value.to(self.device_) for key, value in batch.items()}

        prediction = self(batch)
        prediction = prediction.view(-1, self.lookahead, self.outdim)
        ade, fde = compute_ade_fde(prediction, batch["target"])

        loss = self.criterion(prediction, batch["target"])

        self.log("val_loss", loss, on_epoch=True, on_step=False)
        self.log("val_ade", ade.mean(), on_epoch=True, on_step=False)
        self.log("val_fde", fde.mean(), on_epoch=True, on_step=False)

        return loss

    def test_step(self, batch, batch_idx):
        batch = {key: value.to(self.device_) for key, value in batch.items()}

        prediction = self(batch)
        prediction = prediction.view(-1, self.lookahead, self.outdim)
        ade, fde = compute_ade_fde(prediction, batch["target"])

        loss = self.criterion(prediction, batch["target"])

        self.log("val_loss", loss, on_epoch=True, on_step=False)
        self.log("val_ade", ade.mean(), on_epoch=True, on_step=False)
        self.log("val_fde", fde.mean(), on_epoch=True, on_step=False)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=5e-4,
            weight_decay=1e-5,
        )
        # Warm-up + Cosine Annealing Scheduler
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, 
            lr_lambda=lambda epoch: min((epoch + 1) / 10, 1.0)  # Warm-up for 10 epochs
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}