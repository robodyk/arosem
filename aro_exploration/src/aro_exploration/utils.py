import numpy as np
import tf.transformations as tft
from nav_msgs.msg import MapMetaData

def map_to_grid_coordinates(map_pos: np.ndarray, grid_info: MapMetaData) -> np.ndarray:
    """ convert the input position to the discrete coordinates of a corresponding grid cell """
    # map_pos is (2,) numpy array specifying a continuous position in /icp_map frame
    grid_pos = np.array([0, 0])

    # TODO implement the conversion, don't forget the grid can be both translated and rotated

    # TODO use the grid origin pose (pose of grid's bottom left corner in /icp_map frame) and cell resolution from grid_info

    # TODO return a numpy array of [cell_x, cell_y]  (see image at courseware for more indexing info)

    # TODO note that grid is indexed as [row, column], so occupancy values should be accesed as occupancy_grid[cell_y, cell_x]


    return grid_pos


def grid_to_map_coordinates(grid_pos: np.ndarray, grid_info: MapMetaData) -> np.ndarray:
    """ convert grid cell coordinates to a position in the /icp_map frame """
    # grid_pos is (2,) numpy array of [cell_x, cell_y] specifying a cell index in the grid's x and y directions
    map_pos = np.array([0, 0])

    # TODO implement the conversion, don't forget the grid can be both translated and rotated

    # TODO use the grid origin pose (pose of grid's bottom left corner in /icp_map frame) and cell resolution from grid_info

    # TODO return a numpy array of [x, y] specifying a position in /icp_map frame

    # TODO the output position should correspond to the center of the input cell


    return map_pos

def get_circular_dilation_footprint(robot_diameter, grid_resolution):
    """ Returns a binary mask for inflating obstacles and unknown space, corresponding to circular robot """
    dilate_size = np.ceil((robot_diameter / grid_resolution) / 2).astype(int)
    kernel_width = 2 * dilate_size + 1
    kernel = np.zeros((kernel_width,kernel_width), dtype=bool)
    y, x = np.ogrid[-dilate_size:dilate_size + 1, -dilate_size:dilate_size + 1]
    mask = x ** 2 + y ** 2 <= dilate_size ** 2
    kernel[mask] = 1
    return kernel

