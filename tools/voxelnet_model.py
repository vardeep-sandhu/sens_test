import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F
from loss import *
from voxelnet_point_sampler import VoxelFeatureExtractor,BEVConverter, TrajectoryPredictionModel, VoxelGridProcessor, VoxelBackbone3D, TrajectoryHead3D

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

        self.bev_converter = BEVConverter(3, "max")
        self.trajectory_head = TrajectoryPredictionModel(3)
        self.criterion = trajectory_loss
        self.device_ = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def extract_feat(self, data):
        input_features = self.reader(data["voxels"], data['num_points'], data["mask"])
        return input_features
    
    def grid_action(self, data, voxel_features):
        B, N, output_dim = voxel_features.shape
        H, W, D = map(int, data['grid_shape'])

        # Initialize a grid with zeros
        grid = torch.zeros((B, output_dim, D, W, H), device=voxel_features.device)

        for b in range(B):
            # Select valid voxels for this batch sample
            valid_mask = data["mask"][b]  # Shape: (N,)
            valid_features = voxel_features[b][valid_mask]  # Shape: (num_valid_voxels, output_dim)
            valid_coordinates = data['voxel_coordinates'][b][valid_mask].long()  # Shape: (num_valid_voxels, 3)

            # Scatter valid features into the grid
            for i in range(output_dim):
                grid[b, i, valid_coordinates[:, 0], valid_coordinates[:, 1], valid_coordinates[:, 2]] = valid_features[:, i]
        return grid
    
    def forward(self, data):
        encoded_features = self.extract_feat(
            data
        ) # B, N, 3

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