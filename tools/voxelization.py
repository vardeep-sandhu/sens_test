from ops import points_to_voxel
import numpy as np 
class Voxelization:
    def __init__(self, voxel_size, point_cloud_range, max_num_points, max_voxels):
        self.range = point_cloud_range
        self.voxel_size = voxel_size
        self.max_points_in_voxel = max_num_points
        self.max_voxel_num = max_voxels
        grid_size = (point_cloud_range[3:] - point_cloud_range[:3]) / voxel_size
        self.grid_size = np.round(grid_size).astype(np.int64)

    def voxelize(self, data):
        voxels, coordinates, num_points = points_to_voxel(data["points"], self.voxel_size, self.range, self.max_points_in_voxel, True, self.max_voxel_num)
        num_voxels = np.array([voxels.shape[0]], dtype=np.int64)

        data["voxels"] = dict(
            voxels=voxels,
            coordinates=coordinates,
            num_points=num_points,
            num_voxels=num_voxels,
            shape=self.grid_size,
            range=self.range,
            size=self.voxel_size
        )
        return data