from pathlib import Path

import numpy as np
import torch
from torch.utils.data import  Dataset

import os
from scipy.spatial.transform import Rotation as R
from voxelization import Voxelization

class SemanticKITTIDataset(Dataset):
    def __init__(self, data_path: str, lookahead: int = 30, train=True):
        self.data_path = Path(data_path)
        self.lookahead = lookahead
        self.train = train
        self.train_poses = self._load_poses()[:4000]
        self.test_poses = self._load_poses()[4000:]
        self.train_lidar_files = sorted((self.data_path / "velodyne").glob("*.bin"))[
            :4000
        ]
        self.test_lidar_files = sorted((self.data_path / "velodyne").glob("*.bin"))[
            4000:
        ]
        prange = np.array([-50, -50, -1, 50, 50, 3])
        voxel_size= np.array([0.2, 0.2, 0.3])
        max_points_in_voxel = 5
        max_voxel_num = 150000
        
        self.voxel_generator = Voxelization(voxel_size, prange, max_points_in_voxel, max_voxel_num)
    
    def _load_poses(self) -> np.ndarray:
        calib = self._parse_calibration(self.data_path / "calib.txt")
        return self._parse_poses(self.data_path / "poses.txt", calib)

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
    
    def do_voxelization(self, data):
        return self.voxel_generator.voxelize(data)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        data = {}
        if self.train:
            lidar_data = self._load_lidar(self.train_lidar_files[idx])
            current_pose = self.train_poses[idx]
            target_poses = self.train_poses[idx + 1 : idx + 1 + self.lookahead]
        else:
            lidar_data = self._load_lidar(self.test_lidar_files[idx])
            current_pose = self.test_poses[idx]
            target_poses = self.test_poses[idx + 1 : idx + 1 + self.lookahead]

        # Convert lidar data to tensor
        lidar_tensor = torch.from_numpy(lidar_data).float()

        # Compute relative poses for all lookahead steps
        relative_poses = []
        for target_pose in target_poses:
            relative_pose = np.matmul(np.linalg.inv(current_pose), target_pose)
            translation = relative_pose[:3, 3]
            rotation = relative_pose[:3, :3]

            quaternion = R.from_matrix(
                rotation
            ).as_quat()  # Returns [q_x, q_y, q_z, q_w]
            # Combine translation and rotation into a single tensor
            pose_tensor = torch.cat(
                [
                    torch.from_numpy(translation).float(),
                    torch.from_numpy(quaternion).float(),
                ]
            )
            relative_poses.append(pose_tensor)

        # Stack all relative poses into a single tensor
        data["points"] = lidar_data

        data = self.do_voxelization(data)
        data["target"] = torch.stack(relative_poses)
        return data