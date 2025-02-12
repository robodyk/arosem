"""Library for easier working with point clouds."""
from typing import List, Tuple

import numpy as np
from enum import Enum
from numpy.lib.recfunctions import structured_to_unstructured, unstructured_to_structured


class Channels(Enum):
    """Various channels the pointcloud can consist of. Some channels are "logical" and represent multiple "real"
    channels."""
    position = ['x', 'y', 'z']
    mean = ['mean_%s' % f for f in 'xyz']
    normal = ['normal_%s' % f for f in 'xyz']
    curvature = ['curvature']
    roughness = ['roughness']
    viewpoint = ['vp_%s' % f for f in 'xyz']
    projection = ['proj_%s' % f for f in 'xyz']


position_dtype = [(f, 'f4') for f in Channels.position.value]
mean_dtype = [(f, 'f4') for f in Channels.mean.value]
normal_dtype = [(f, 'f4') for f in Channels.normal.value]
curvature_dtype = [(f, 'f4') for f in Channels.curvature.value]
roughness_dtype = [(f, 'f4') for f in Channels.roughness.value]
viewpoint_dtype = [(f, 'f4') for f in Channels.viewpoint.value]
projection_dtype = [(f, 'f4') for f in Channels.projection.value]


class DType(Enum):
    """Datatypes of channels to be used in structured Numpy arrays."""
    position = np.dtype(position_dtype)
    mean = np.dtype(mean_dtype)
    normal = np.dtype(normal_dtype)
    curvature = np.dtype(curvature_dtype)
    roughness = np.dtype(roughness_dtype)
    viewpoint = np.dtype(viewpoint_dtype)
    projection = np.dtype(projection_dtype)


def position(x_struct: np.ndarray) -> np.ndarray:
    """Return the x/y/z positions of points in the structured pointcloud.
    :param x_struct: The structured pointcloud. Numpy structured array N.
    :return: The x/y/z positions of points in the structured pointcloud. Numpy array Nx3.
    """
    return structured_to_unstructured(x_struct[Channels.position.value])


def normal(x_struct: np.ndarray) -> np.ndarray:
    """Return the normals of points in the structured pointcloud.
    :param x_struct: The structured pointcloud. Numpy structured array N.
    :return: The normals of points in the structured pointcloud. Numpy array Nx3.
    """
    return structured_to_unstructured(x_struct[Channels.normal.value])


def viewpoint(x_struct: np.ndarray) -> np.ndarray:
    """The viewpoints of points in the structured pointcloud (i.e. sensor optical centers from times of observation).
    :param x_struct: The structured pointcloud. Numpy structured array N.
    :return: The viewpoints of points in the structured pointcloud. Numpy array Nx3.
    """
    return structured_to_unstructured(x_struct[Channels.viewpoint.value])


def e2p(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Convert a 3D Euclidean vector to a 4D homogeneous vector.
    :param x: The 3D Euclidean vector.
    :param axis: The axis along which the 4th dimension should be added.
    :return: The 4D homogeneous vector.
    """
    h_size = list(x.shape)
    h_size[axis] = 1
    h = np.ones(h_size, dtype=x.dtype)
    xh = np.concatenate((x, h), axis=axis)
    return xh


def p2e(xh: np.ndarray, axis: int = -1) -> np.ndarray:
    """Convert a 4D homogeneous vector to a 3D Euclidean vector.
    :param xh: The 4D homogeneous vector.
    :param axis: The axis from which the 4th dimension should be removed.
    :return: The 3D Euclidean vector.
    """
    if axis != -1:
        xh = xh.swapaxes(axis, -1)
    x = xh[..., :-1]
    if axis != -1:
        x = x.swapaxes(axis, -1)
    return x


def transform(T: np.ndarray, x_struct: np.ndarray) -> np.ndarray:
    """Transform a structured point cloud by the given homogeneous transformation matrix, correctly accounting for the
    fact that some channels need to be transformed by rotation and translation and some only by rotation.
    :param T: The homogeneous transformation matrix. Numpy array 4x4.
    :param x_struct: The structured point cloud. Numpy structured array N.
    :return: The transformed structured point cloud. Numpy structured array N.
    """
    assert T.shape == (4, 4)
    x_struct = x_struct.copy()
    fields_op: List[Tuple[List[str], str]] = []
    for fs, dtype, op in (
            (Channels.position.value, DType.position.value, 'Rt'),
            (Channels.viewpoint.value, DType.viewpoint.value, 'Rt'),
            (Channels.normal.value, DType.normal.value, 'R')):
        if fs[0] in x_struct.dtype.fields:
            fields_op.append((fs, op))
    for fs, op in fields_op:
        x = structured_to_unstructured(x_struct[fs])
        if op == 'Rt':
            x = p2e(e2p(x) @ T.T)
        elif op == 'R':
            x = x @ T[:-1, :-1].T
        x_str = unstructured_to_structured(x, dtype=dtype)
        for i in range(len(fs)):
            x_struct[fs[i]] = x_str[dtype.names[i]]
    return x_struct
