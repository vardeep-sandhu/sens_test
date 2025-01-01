from ops import points_to_voxel
import numpy as np 

class Voxelization:
    def __init__(self, config):
        self.range = np.array(config.point_cloud_range)
        self.voxel_size = np.array(config.voxel_size)
        self.max_points_in_voxel = config.max_points_in_voxel
        self.max_voxel_num = config.max_voxel_num
        grid_size = (self.range[3:] - self.range[:3]) / self.voxel_size
        self.grid_size = np.round(grid_size).astype(np.int64)

    def voxelize(self, points):
        voxels, coordinates, num_points = points_to_voxel(points, self.voxel_size, self.range, self.max_points_in_voxel, True, self.max_voxel_num)
        num_voxels = np.array([voxels.shape[0]], dtype=np.int64)

        return dict(
            voxels=voxels,
            coordinates=coordinates,
            num_points=num_points,
            num_voxels=num_voxels,
            shape=self.grid_size,
            range=self.range,
            size=self.voxel_size
        )