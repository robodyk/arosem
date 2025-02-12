from typing import Tuple, Optional, Sequence, Union

import numpy as np


class VoxelMap:
    """Voxel map implementation."""
    
    voxel_size: float
    """Size of the voxels in meters."""
    
    free_update: float
    """Value by which a free cell should be marked when inserting lines."""
    
    hit_update: float
    """Value by which a hit cell should be marked when inserting lines."""
    
    occupied_threshold: float
    """Threshold for marking a cell as occupied."""

    def __init__(self, voxel_size: float, free_update: float, hit_update: float, occupied_threshold: float):
        """
        :param voxel_size: Size of the voxels in meters.
        :param free_update: Value by which a free cell should be marked when inserting lines.
        :param hit_update: Value by which a hit cell should be marked when inserting lines.
        :param occupied_threshold: Threshold for marking a cell as occupied.
        """
        pass

    def update_lines(self, x: np.ndarray, y: np.ndarray) -> None:
        """Update the map by marking endpoints from y as occupied and the rest of rays from x as free.
        :param x: Starts of the lines. Numpy array 3xN.
        :param y: Ends of the lines. Numpy array 3xN.
        """
        pass
    
    def get_voxels(self, x: Optional[np.ndarray] = None, l: Optional[Sequence[float]] = None) -> \
            Union[Tuple[np.ndarray, np.ndarray, np.ndarray], np.ndarray]:
        """Get all defined voxels.
        
        :param x: Optional preallocated array into which the voxel coordinates will be stored. Numpy array 3xN.
        :param l: Optional preallocated array into which the levels will be stored. Array-like N.
        :return: If `x` and `l` are not specified, returns 3-tuple (x, l, v), otherwise returns just v. v are voxel
                 values, Numpy array N.
        """
        pass
    
    def set_voxels(self, x: np.ndarray, l: np.ndarray, v: np.ndarray) -> None:
        """Set the given voxels to the given values.
        
        :param x: Voxel coordinates. Numpy array 3xN.
        :param l: Levels of the voxel coordinates. Numpy array N.
        :param v: Values at the voxel coordinates. Numpy array N.
        """
        pass
    
    def update_voxels(self, x: np.ndarray, l: np.ndarray, v: np.ndarray) -> None:
        """Update the given voxels with the given values (add to the already present values).
        
        :param x: Voxel coordinates. Numpy array 3xN.
        :param l: Levels of the voxel coordinates. Numpy array N.
        :param v: Values at the voxel coordinates. Numpy array N.
        """
        pass
    
    def size(self) -> int: ...
    def clear(self) -> None: ...
        