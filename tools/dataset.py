from pathlib import Path

import numpy as np
import torch
from torch.utils.data import  Dataset

from scipy.spatial.transform import Rotation as R
from voxelization import Voxelization
from augmentations import *

class SemanticKITTIDataset(Dataset):
    def __init__(self, config: dict, train=True):
        self.data_path = Path(config.dataset_path)
        self.lookahead = config.lookahead
        self.train = train

        # Load poses and lidar files
        self.train_poses, self.test_poses = self._split_data(self._load_poses())
        self.train_lidar_files, self.test_lidar_files = self._split_data(
            sorted((self.data_path / "velodyne").glob("*.bin"))
        )

        self.voxel_generator = Voxelization(config.voxelization)
    
    def _load_poses(self) -> np.ndarray:
        calib = self._parse_calibration(self.data_path / "calib.txt")
        return self._parse_poses(self.data_path / "poses.txt", calib)

    def _split_data(self, data):
        split_idx = 4000
        return data[:split_idx], data[split_idx:]

    @staticmethod
    def _parse_calibration(filename: Path) -> dict[str, np.ndarray]:
        calib = {}
        for line in filename.read_text().splitlines():
            key, content = line.strip().split(":")
            values = [float(v) for v in content.strip().split()]
            pose = np.zeros((4, 4))
            pose[0, 0:4] = values[0:4]
            pose[1, 0:4] = values[4:8]
            pose[2, 0:4] = values[8:12]
            pose[3, 3] = 1.0
            calib[key] = pose
        return calib

    @staticmethod
    def _parse_poses(filename: Path, calibration: dict[str, np.ndarray]) -> np.ndarray:
        poses = []
        cab_tr = calibration["Tr"]
        tr_inv = np.linalg.inv(cab_tr)
        for line in filename.read_text().splitlines():
            values = [float(v) for v in line.strip().split()]
            pose = np.zeros((4, 4))
            pose[0, 0:4] = values[0:4]
            pose[1, 0:4] = values[4:8]
            pose[2, 0:4] = values[8:12]
            pose[3, 3] = 1.0
            poses.append(np.matmul(tr_inv, np.matmul(pose, cab_tr), dtype=np.float32))
        return np.array(poses, dtype=np.float32)

    @staticmethod
    def _load_lidar(lidar_file: Path) -> np.ndarray:
        scan = np.fromfile(lidar_file, dtype=np.float32).reshape((-1, 4))
        return scan[:, :3]

    def __len__(self) -> int:
        if self.train:
            return len(self.train_lidar_files) - (self.lookahead + 1)
        else:
            return len(self.test_lidar_files) - (self.lookahead + 1)
        

    def _get_data_split(self, idx: int):
        """Gets the appropriate data split (train or test) based on the mode."""
        if self.train:
            lidar_files, poses = self.train_lidar_files, self.train_poses
        else:
            lidar_files, poses = self.test_lidar_files, self.test_poses

        lidar_data = self._load_lidar(lidar_files[idx])
        current_pose = poses[idx]
        target_poses = poses[idx + 1 : idx + 1 + self.lookahead]
        return lidar_data, current_pose, target_poses

    def _compute_relative_poses(self, current_pose, target_poses):
        """Computes relative poses between current and target poses."""
        relative_poses = []
        for target_pose in target_poses:
            relative_pose = np.matmul(np.linalg.inv(current_pose), target_pose)
            translation = relative_pose[:3, 3]
            rotation = relative_pose[:3, :3]
            quaternion = R.from_matrix(rotation).as_quat()  # Returns [q_x, q_y, q_z, q_w]

            pose_tensor = torch.cat(
                [
                    torch.from_numpy(translation).float(),
                    torch.from_numpy(quaternion).float(),
                ]
            )
            relative_poses.append(pose_tensor)
        return torch.stack(relative_poses)


    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        data = {}
        lidar_data, current_pose, target_poses = self._get_data_split(idx)
        if self.train:
            lidar_data = self.augment_point_cloud(lidar_data)

        relative_poses = self._compute_relative_poses(current_pose, target_poses)        
        voxels = self.do_voxelization(lidar_data)

        data = {"points" : lidar_data, 
                "voxels" : voxels,
                "target" : relative_poses
        }
        return data

    def do_voxelization(self, data):
        """Applies voxelization to the point cloud data."""
        return self.voxel_generator.voxelize(data)

    def augment_point_cloud(self, point_cloud):
        """Applies augmentations to the point cloud."""
        augmentations = [random_rotation, random_scaling, random_translation, random_jittering]
        for aug in augmentations:
            point_cloud = aug(point_cloud)
        return point_cloud
