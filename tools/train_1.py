from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

import torch.nn.functional as F
import spconv.pytorch as spconv
import pytorch_lightning as pl
import os
from dataset import SemanticKITTIDataset
from tools.crude_voxel_downsampling import OptimizedTrajectoryPredictor
from lightning.pytorch.loggers import TensorBoardLogger
from voxelnet_model import TrajectoryPredictorWithVoxelNet


def custom_collate_fn(batch):
    """
    Custom collate function to handle variable-sized point clouds in a batch.

    Parameters:
        batch (list): List of (point_cloud, label) tuples.
                      - point_cloud: Tensor of shape (N, 3)
                      - label: Tensor or scalar for ground truth.

    Returns:
        padded_point_clouds: Tensor of shape (B, max_N, 3)
        masks: Tensor of shape (B, max_N) indicating valid points.
        labels: Tensor of shape (B, *) ground truth labels.
    """
    # Separate point clouds and labels
    point_clouds, labels = zip(*batch)

    # Find max number of points in the batch
    max_points = max(pc.shape[0] for pc in point_clouds)

    # Pad point clouds to max_points
    padded_point_clouds = torch.zeros((len(batch), max_points, 3), dtype=torch.float32)
    masks = torch.zeros((len(batch), max_points), dtype=torch.bool)

    for i, pc in enumerate(point_clouds):
        n_points = pc.shape[0]
        padded_point_clouds[i, :n_points] = pc
        masks[i, :n_points] = True

    # Convert labels to a tensor
    labels = (
        torch.stack(labels)
        if isinstance(labels[0], torch.Tensor)
        else torch.tensor(labels)
    )

    return padded_point_clouds, masks, labels


# Additional Training Configuration
def train_model():
    logger = TensorBoardLogger("tb_logs", name="first_model")
    # Lightning Trainer with optimizations
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,  # Or multiple GPUs
        precision=16,  # 16-bit precision
        max_epochs=100,
        logger=logger,
        # gradient_clip_val=0.5,
        # accumulate_grad_batches=4,  # Simulate larger batch sizes
        strategy="auto",
    )

    # Initialize model
    model = TrajectoryPredictorWithVoxelNet(
        voxel_size=0.2, grid_bounds=((-5, 5), (-5, 5), (-2, 2))
    )

    # Create dataset and dataloader
    train_dataset = SemanticKITTIDataset(
        "/home/sandhu/learning/sensmore_test/SemanticKITTI_00", train=True
    )
    test_dataset = SemanticKITTIDataset(
        "/home/sandhu/learning/sensmore_test/SemanticKITTI_00", train=False
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=2,
        num_workers=os.cpu_count(),
        pin_memory=True,
        collate_fn=custom_collate_fn,
        persistent_workers=True,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=2,
        num_workers=os.cpu_count(),
        pin_memory=True,
        collate_fn=custom_collate_fn,
        persistent_workers=True,
    )

    # Train the model
    trainer.fit(model, train_dataloader, test_dataloader)
    # trainer.test(
    #     model,
    #     test_dataloader,
    #     ckpt_path="/home/sandhu/learning/sensmore_test/tb_logs/first_model/version_42/checkpoints/epoch=0-step=1985.ckpt",
    # )


train_model()
