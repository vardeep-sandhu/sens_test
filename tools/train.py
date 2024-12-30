import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import os
from dataset import SemanticKITTIDataset
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
    voxels = [torch.Tensor(i["voxels"]["voxels"]) for i in batch]
    max_num_voxels = max(v.shape[0] for v in voxels)
    num_points = [torch.Tensor(i["voxels"]["num_points"]) for i in batch]
    voxel_coordinates = [torch.Tensor(i['voxels']['coordinates']) for i in batch]
    grid_shape = torch.Tensor(batch[0]['voxels']['shape'])
    targets = [i["target"] for i in batch]
    voxel_shape = voxels[0].shape[1:]
    padded_voxels = []
    masks = []
    padded_num_points = []
    padded_voxel_coordinates = []

    for i, _ in enumerate(voxels):
        num_voxels = voxels[i].shape[0]
        
        padded = torch.zeros((max_num_voxels, *voxel_shape), dtype=voxels[i].dtype)
        p_num_points = torch.zeros((max_num_voxels))
        p_voxel_coordinates = torch.zeros((max_num_voxels, 3))
        p_voxel_coordinates[:num_voxels] = voxel_coordinates[i]
        p_num_points[:num_voxels] = num_points[i]
        padded[:num_voxels] = voxels[i]  # Copy valid voxels
        padded_voxels.append(padded)
        padded_num_points.append(p_num_points)
        padded_voxel_coordinates.append(p_voxel_coordinates)
        
        mask = torch.zeros(max_num_voxels, dtype=torch.bool)
        mask[:num_voxels] = True
        masks.append(mask)
    batch_voxels = torch.stack(padded_voxels)
    batch_masks = torch.stack(masks)
    batch_num_points = torch.stack(padded_num_points)  # Optional
    batch_voxel_coordinates = torch.stack(padded_voxel_coordinates)
    return {
        'voxels': batch_voxels,
        'mask': batch_masks,
        'num_points': batch_num_points,
        'grid_shape': grid_shape,
        'voxel_coordinates' : batch_voxel_coordinates,
        'target': torch.stack(targets)
    }

def train_model(dataset_path: str, do_test: bool):
    logger = TensorBoardLogger("tb_logs", name="first_model")
    trainer = pl.Trainer(
        accelerator="cpu",
        devices=1,
        precision=16,
        max_epochs=10,
        logger=logger,
        strategy="auto",
    )
    model = TrajectoryPredictorWithVoxelNet()

    train_dataset = SemanticKITTIDataset(dataset_path, train=True)
    test_dataset = SemanticKITTIDataset(dataset_path, train=False)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=2,
        num_workers=os.cpu_count(),
        pin_memory=True,
        shuffle=True,
        collate_fn=custom_collate_fn,
        persistent_workers=True,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=2,
        num_workers=os.cpu_count(),
        pin_memory=True,
        shuffle=False,
        collate_fn=custom_collate_fn,
        persistent_workers=True,
    )
    trainer.fit(model, train_dataloader, test_dataloader)
    if do_test:
        trainer.test(
            model,
            test_dataloader,
            ckpt_path="/home/sandhu/learning/sensmore_test/tb_logs/first_model/version_42/checkpoints/epoch=0-step=1985.ckpt",
        )

if __name__ == "__main__":
    do_test = False
    DATASET = "/home/sandhu/project/sens_test/SemanticKITTI_00"
    train_model(dataset_path=DATASET, do_test=do_test)
