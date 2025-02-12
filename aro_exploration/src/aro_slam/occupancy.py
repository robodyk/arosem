"""Construct occupancy map from sequentially coming pointclouds."""

import numpy as np
import rospy

from geometry_msgs.msg import Point
from nav_msgs.msg import OccupancyGrid
# In Pycharm, right-click folder voxel_map_stubs, Mark directory as->Sources root to get documentation for the functions
from voxel_map import VoxelMap

from aro_slam.utils import array, col, logistic


class OccupancyMap:
    """Construct occupancy map from sequentially coming pointclouds."""

    def __init__(self, frame_id: str, resolution=0.1, empty_update=-1.0, occupied_update=1.0, min=-10.0, max=10.0,
                 occupied=5.0):
        """
        :param frame_id: ROS TF frame ID of the map.
        :param resolution: Resolution of map cells in meters.
        :param empty_update: Value by which empty cells are updated when adding a new scan.
        :param occupied_update: Value by which hit cells are updated when adding a new scan.
        :param min: Minimum cell value (all values will be clipped at this value).
        :param max: Maximum cell value (all values will be clipped at this value).
        :param occupied: Threshold for values to mark cells as occupied.
        """
        self.voxel_map = VoxelMap(resolution, empty_update, occupied_update, occupied)
        self.min = min
        self.max = max
        self.msg = OccupancyGrid()
        self.msg.header.frame_id = frame_id
        self.msg.info.resolution = resolution
        # Initially, set the grid origin to identity.
        self.msg.info.origin.orientation.w = 1.0

    def map_to_grid(self, x: np.ndarray) -> np.ndarray:
        """Transform points from map coordinates to grid.
        :param x: Points in map coordinates.
        :return: Points in grid coordinates.
        """
        x = x - col(array(self.msg.info.origin.position))
        return x

    def grid_to_map(self, x: np.ndarray) -> np.ndarray:
        """Transform points from grid coordinates to map.
        :param x: Points in grid coordinates.
        :return: Points in map coordinates.
        """
        x = x + col(array(self.msg.info.origin.position))
        return x

    def fit_grid(self) -> None:
        """Accommodate the grid to contain all points."""
        # Update grid origin so that all coordinates are non-negative.
        x, _, v = self.voxel_map.get_voxels()
        if x.size == 0:
            return
        x = x[:2]  # Only x,y used in 2D grid.
        x_min = x.min(axis=1) - self.voxel_map.voxel_size / 2.
        x_max = x.max(axis=1) + self.voxel_map.voxel_size / 2.
        nx = np.round((x_max - x_min) / self.msg.info.resolution).astype(np.int)
        self.msg.info.origin.position = Point(x_min[0], x_min[1], 0.0)
        self.msg.info.width, self.msg.info.height = nx

    def grid_voxels(self) -> np.ndarray:
        """Return voxel coordinates corresponding to the current grid.
        :return: Voxel center coordinates (in grid coordinates).
        """
        i, j = np.meshgrid(np.arange(self.msg.info.width), np.arange(self.msg.info.height), indexing='xy')
        x = np.stack((i.ravel(), j.ravel(), np.zeros_like(i).ravel()))
        x = (x + 0.5) * self.msg.info.resolution
        return x

    def to_msg(self) -> OccupancyGrid:
        """Return the grid as ROS message. Update grid parameters as needed.
        :return: The grid as ROS message.
        """
        self.fit_grid()
        x = self.grid_voxels()
        x = self.grid_to_map(x)
        x[2, :] = self.voxel_map.voxel_size / 2.0
        l = np.zeros((x.shape[1],))
        v = self.voxel_map.get_voxels(x, l)
        v = 100. * logistic(v)
        v[np.isnan(v)] = -1.
        self.msg.data = v.astype(int).tolist()
        return self.msg

    def voxel_map_points(self, x: np.ndarray) -> np.ndarray:
        """Get corresponding voxels to the given pointcloud (project to 2.5D map).
        :param x: Pointcloud. Numpy array 3xN.
        :return: Pointcloud projected to the 2.5D grid (i.e., z coordinate will be constant). Numpy array 3xN.
        """
        x = x.copy()
        x[2, :] = self.voxel_map.voxel_size / 2.0
        return x

    def update(self, x: np.ndarray, y: np.ndarray, stamp: rospy.Time) -> None:
        """Update internal occupancy map by inserting rays.
        :param x: Starting points of rays. Numpy array 3xN.
        :param y: End points of rays. Numpy array 3xN. Must have same size as x.
        :param stamp: Timestamp of the update.
        """
        x = self.voxel_map_points(x)
        y = self.voxel_map_points(y)
        if x.shape[1] == 1:
            x = np.broadcast_to(x, y.shape)
        elif y.shape[1] == 1:
            y = np.broadcast_to(y, x.shape)
        self.voxel_map.update_lines(x, y)
        self.clip_values()
        self.msg.header.stamp = stamp
        self.msg.info.map_load_time = stamp

    def occupied(self, x: np.ndarray) -> np.ndarray:
        """Occupied flags for points x."""
        x = self.voxel_map_points(x)
        l = np.zeros((x.shape[1],))
        v = self.voxel_map.get_voxels(x, l)
        occupied = v > self.voxel_map.occupied_threshold
        return occupied

    def clip_values(self) -> None:
        """Clip values between min and max."""
        x, l, v = self.voxel_map.get_voxels()
        v = np.clip(v, self.min, self.max)
        self.voxel_map.set_voxels(x, l, v)
