from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

class SemanticKITTIDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        lookahead: int = 30,
    ):
        self.data_path = Path(data_path)
        self.lookahead = lookahead

        self.poses = self._load_poses()
        self.lidar_files = sorted((self.data_path / "velodyne").glob("*.bin"))

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
        return len(self.lidar_files) - self.lookahead

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        lidar_data = self._load_lidar(self.lidar_files[idx])

        # Convert to tensor
        lidar_tensor = torch.from_numpy(lidar_data).float()

        # Calculate relative pose transformation
        current_pose = self.poses[idx]
        target_pose = self.poses[idx + self.lookahead]

        # Compute relative transformation
        relative_pose = np.matmul(np.linalg.inv(current_pose), target_pose)

        # Extract translation and rotation
        translation = relative_pose[:3, 3]
        rotation = relative_pose[:3, :3]

        # Combine into target tensor
        target = torch.cat(
            [
                torch.from_numpy(translation).float(),
                torch.from_numpy(rotation.flatten()).float(),
            ]
        )
        return lidar_tensor, target


class TrajectoryPredictor(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=128, output_dim=12):
        super().__init__()
        # Point cloud feature extraction
        self.point_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.LayerNorm(hidden_dim)
        )

        # Transformer encoder for sequence modeling
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_dim, nhead=8, dim_feedforward=256, dropout=0.1
            ),
            num_layers=3,
        )

        # Trajectory prediction head
        self.trajectory_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, num_points, 3)

        # Encode point cloud features
        point_features = self.point_encoder(x)

        # Transformer expects (seq_len, batch_size, features)
        point_features = point_features.permute(1, 0, 2)
        print(point_features.shape)

        # Apply transformer
        encoded_features = self.transformer_encoder(point_features)

        # Global pooling
        global_feature = encoded_features.mean(dim=0)

        # Predict trajectory
        trajectory = self.trajectory_head(global_feature)

        return trajectory


def train(model: nn.Module, train_loader: DataLoader, num_epochs: int):
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    device = torch.device(device)

    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # Adjust to your needs
    criterion = nn.MSELoss()  # Adjust to your needs
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0

        for lidar_points, target in train_loader:
            lidar_points = lidar_points.to(device)
            target = target.to(device)

            optimizer.zero_grad()

            # Predict trajectory
            prediction = model(lidar_points)

            # Compute loss (e.g., MSE for translation and rotation)
            translation_loss = criterion(prediction[:, :3], target[:, :3])
            rotation_loss = criterion(prediction[:, 3:], target[:, 3:])

            loss = translation_loss + rotation_loss

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            bar.set_description(
                f"Epoch {epoch+1:2}/{num_epochs} Epoch Loss: {total_loss/len(train_loader):.4f} Loss: {loss:.2f}"
            )

    return model


def main():
    dataset = SemanticKITTIDataset(
        "/home/sandhu/learning/sensmore_test/SemanticKITTI_00"
    )
    train_loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=True,
        num_workers=2,  # Adjust this to your machine
    )
    # train_loader[1]
    model = TrajectoryPredictor()
    # train(model, train_loader, num_epochs=50)


if __name__ == "__main__":
    main()
