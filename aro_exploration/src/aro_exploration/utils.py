import numpy as np
import tf.transformations as tft
from nav_msgs.msg import MapMetaData


def map_to_grid_coordinates(map_pos: np.ndarray, grid_info: MapMetaData) -> np.ndarray:
    """Convert the input position to the discrete coordinates of a corresponding grid cell"""
    # map_pos is (2,) numpy array specifying a continuous position in /icp_map frame

    # Extract origin position and orientation from grid_info
    origin_pos = np.array([grid_info.origin.position.x, grid_info.origin.position.y])
    origin_quat = [
        grid_info.origin.orientation.x,
        grid_info.origin.orientation.y,
        grid_info.origin.orientation.z,
        grid_info.origin.orientation.w,
    ]

    # Convert quaternion to rotation matrix
    rot_matrix = tft.quaternion_matrix(origin_quat)[:2, :2]

    # Calculate position relative to grid origin
    rel_pos = map_pos - origin_pos

    # Apply inverse rotation to get position in grid's coordinate system
    # Transpose of rotation matrix is its inverse for orthogonal matrices
    grid_aligned_pos = np.dot(rot_matrix.T, rel_pos)

    # Convert to cell indices by dividing by cell resolution
    # We floor the result to get the cell index
    cell_indices = np.floor(grid_aligned_pos / grid_info.resolution).astype(int)

    # Return as [cell_x, cell_y]
    return cell_indices


def grid_to_map_coordinates(grid_pos: np.ndarray, grid_info: MapMetaData) -> np.ndarray:
    """Convert grid cell coordinates to a position in the /icp_map frame"""
    # grid_pos is (2,) numpy array of [cell_x, cell_y] specifying a cell index in the grid's x and y directions

    # Extract origin position and orientation from grid_info
    origin_pos = np.array([grid_info.origin.position.x, grid_info.origin.position.y])
    origin_quat = [
        grid_info.origin.orientation.x,
        grid_info.origin.orientation.y,
        grid_info.origin.orientation.z,
        grid_info.origin.orientation.w,
    ]

    # Convert quaternion to rotation matrix
    rot_matrix = tft.quaternion_matrix(origin_quat)[:2, :2]

    # Calculate cell center in grid coordinates
    # Add 0.5 to get the center of the cell
    cell_center = (grid_pos + 0.5) * grid_info.resolution

    # Apply rotation to get position in map coordinate system
    map_aligned_pos = np.dot(rot_matrix, cell_center)

    # Add grid origin offset to get final map position
    map_pos = origin_pos + map_aligned_pos

    return map_pos


def get_circular_dilation_footprint(robot_diameter, grid_resolution):
    """Returns a binary mask for inflating obstacles and unknown space, corresponding to circular robot"""
    dilate_size = np.ceil((robot_diameter / grid_resolution) / 2).astype(int)
    kernel_width = 2 * dilate_size + 1
    kernel = np.zeros((kernel_width, kernel_width), dtype=bool)
    y, x = np.ogrid[-dilate_size : dilate_size + 1, -dilate_size : dilate_size + 1]
    mask = x**2 + y**2 <= dilate_size**2
    kernel[mask] = 1
    return kernel
