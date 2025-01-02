import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import os
from dataset import SemanticKITTIDataset
from lightning.pytorch.loggers import TensorBoardLogger
from voxelnet_model import TrajectoryPredictorWithVoxelNet
import yaml
from box import Box


def load_cofig(path: str) -> Box:
    with open(path, 'r') as file:
        config = yaml.safe_load(file)
    return Box(config)

def custom_collate_fn(batch: list) -> dict:
    voxels = [torch.Tensor(i["voxels"]["voxels"]) for i in batch]
    num_points = [torch.Tensor(i["voxels"]["num_points"]) for i in batch]
    voxel_coordinates = [torch.Tensor(i['voxels']['coordinates']) for i in batch]
    grid_shape = torch.Tensor(batch[0]['voxels']['shape'])
    targets = [i["target"] for i in batch]

    max_num_voxels = max(v.shape[0] for v in voxels)
    voxel_shape = voxels[0].shape[1:]

    padded_voxels, padded_num_points, padded_voxel_coordinates, masks = [], [], [], []

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
    return {
        'voxels': torch.stack(padded_voxels),
        'mask': torch.stack(masks),
        'num_points': torch.stack(padded_num_points),
        'grid_shape': grid_shape,
        'voxel_coordinates': torch.stack(padded_voxel_coordinates),
        'target': torch.stack(targets),
    }


def train_model(config: Box):
    logger = TensorBoardLogger("tb_logs", name="first_model")
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        precision=16,
        max_epochs=config.epochs,
        logger=logger,
        strategy="auto",
    )
    model = TrajectoryPredictorWithVoxelNet()

    train_dataset = SemanticKITTIDataset(config, train=True)
    test_dataset = SemanticKITTIDataset(config, train=False)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.train_batch_size,
        num_workers=os.cpu_count(),
        pin_memory=True,
        shuffle=True,
        collate_fn=custom_collate_fn,
        persistent_workers=True,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=config.test_batch_size,
        num_workers=os.cpu_count(),
        pin_memory=True,
        shuffle=False,
        collate_fn=custom_collate_fn,
        persistent_workers=True,
    )
    trainer.fit(model, train_dataloader, test_dataloader)
    # if config.do_test and config.ckpt_path:
    trainer.test(
        model,
        test_dataloader,
        ckpt_path="best",
    )

# if __name__ == "__main__":
config_path = "/content/drive/MyDrive/Learning/sens_test/config/init_config.yaml"
config = load_cofig(config_path)

train_model(config)
