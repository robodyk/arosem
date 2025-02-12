import math
from typing import Union, List, Sequence, Tuple

import numpy as np

# The following are imports from library tf.transformations included here so that this library can be run without ROS

# epsilon for testing whether a number is close to zero
_EPS = np.finfo(float).eps * 4.0

# axis sequences for Euler angles
_NEXT_AXIS = [1, 2, 0, 1]

# map axes strings to/from tuples of inner axis, parity, repetition, frame
_AXES2TUPLE = {
    'sxyz': (0, 0, 0, 0), 'sxyx': (0, 0, 1, 0), 'sxzy': (0, 1, 0, 0),
    'sxzx': (0, 1, 1, 0), 'syzx': (1, 0, 0, 0), 'syzy': (1, 0, 1, 0),
    'syxz': (1, 1, 0, 0), 'syxy': (1, 1, 1, 0), 'szxy': (2, 0, 0, 0),
    'szxz': (2, 0, 1, 0), 'szyx': (2, 1, 0, 0), 'szyz': (2, 1, 1, 0),
    'rzyx': (0, 0, 0, 1), 'rxyx': (0, 0, 1, 1), 'ryzx': (0, 1, 0, 1),
    'rxzx': (0, 1, 1, 1), 'rxzy': (1, 0, 0, 1), 'ryzy': (1, 0, 1, 1),
    'rzxy': (1, 1, 0, 1), 'ryxy': (1, 1, 1, 1), 'ryxz': (2, 0, 0, 1),
    'rzxz': (2, 0, 1, 1), 'rxyz': (2, 1, 0, 1), 'rzyz': (2, 1, 1, 1)}

_TUPLE2AXES = dict((v, k) for k, v in _AXES2TUPLE.items())


def euler_from_matrix(matrix: Union[Sequence[float], np.ndarray], axes='sxyz') -> Tuple[float, float, float]:
    try:
        firstaxis, parity, repetition, frame = _AXES2TUPLE[axes.lower()]
    except (AttributeError, KeyError):
        _ = _TUPLE2AXES[axes]
        firstaxis, parity, repetition, frame = axes

    i = firstaxis
    j = _NEXT_AXIS[i+parity]
    k = _NEXT_AXIS[i-parity+1]

    M = np.array(matrix, dtype=np.float64, copy=False)[:3, :3]
    if repetition:
        sy = math.sqrt(M[i, j]*M[i, j] + M[i, k]*M[i, k])
        if sy > _EPS:
            ax = math.atan2( M[i, j],  M[i, k])
            ay = math.atan2( sy,       M[i, i])
            az = math.atan2( M[j, i], -M[k, i])
        else:
            ax = math.atan2(-M[j, k],  M[j, j])
            ay = math.atan2( sy,       M[i, i])
            az = 0.0
    else:
        cy = math.sqrt(M[i, i]*M[i, i] + M[j, i]*M[j, i])
        if cy > _EPS:
            ax = math.atan2( M[k, j],  M[k, k])
            ay = math.atan2(-M[k, i],  cy)
            az = math.atan2( M[j, i],  M[i, i])
        else:
            ax = math.atan2(-M[j, k],  M[j, j])
            ay = math.atan2(-M[k, i],  cy)
            az = 0.0

    if parity:
        ax, ay, az = -ax, -ay, -az
    if frame:
        ax, az = az, ax
    return ax, ay, az


# https://stackoverflow.com/a/24837438/1076564
def merge_dicts(dict1, dict2):
    """Recursively merges dict2 into dict1."""
    if not isinstance(dict1, dict) or not isinstance(dict2, dict):
        return dict2
    for k in dict2:
        if k in dict1:
            dict1[k] = merge_dicts(dict1[k], dict2[k])
        else:
            dict1[k] = dict2[k]
    return dict1


def coords_to_transform_matrix(x_y_yaw: Union[Sequence[float], np.ndarray]) -> np.ndarray:
    """Converts the x, y and yaw 2D coordinates to 3D homogeneous transformation matrix.

    :param x_y_yaw: A 3-tuple x, y, yaw either as a Python iterable or as Numpy array 3x1 or 3.
    :return: 3D homogeneous transformation matrix (Numpy array 4x4).
    """
    x, y, yaw = x_y_yaw.ravel().tolist() if isinstance(x_y_yaw, np.ndarray) else x_y_yaw
    c = np.cos(yaw)
    s = np.sin(yaw)
    ma_in_body = np.array([
        [c, -s, 0, x],
        [s,  c, 0, y],
        [0,  0, 1, 0],
        [0,  0, 0, 1],
    ])
    return ma_in_body


def transform_matrix_to_coords(matrix: np.ndarray) -> Tuple[float, float, float]:
    """Convert 4x4 homogeneous 3D transformation matrix to 2D coordinates x, y, yaw.

    :param matrix: The 4x4 transformation matrix (Numpy array 4x4).
    :return: x, y, yaw
    """
    return matrix[0, 3], matrix[1, 3], euler_from_matrix(matrix)[2]


def sec_to_nsec(sec: Union[int, float]) -> int:
    """Return the number in seconds as nanoseconds.

    :param sec: Seconds. Passing large float values will lead to precision loss.
    :return: Nanoseconds.
    """
    return int(sec * int(1e9))


def nsec_to_sec(nsec: Union[int, np.ndarray]) -> Union[int, np.ndarray]:
    """Return the number in nanoseconds as fractional seconds.

    :param nsec: Nanoseconds. Passing too large values (larger than a few days) will lead to precision loss.
    :return: Seconds.
    """
    return nsec / 1e9


def normalize_angle_symmetric(a: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """Normalize the given angle(s) to [-pi, pi].

    :param a: The angle(s) to normalize.
    :return: The normalized angle(s).
    """
    if isinstance(a, np.ndarray) and a.size == 0:
        return a
    return np.mod(a + np.pi, 2 * np.pi) - np.pi


def angular_signed_distance(a1: Union[float, np.ndarray], a2: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """Compute the shorter angular distance(s) between angles `a1` and `a2` such that `a1 + d = a2` (up to wrapping
    around `[-pi, pi]`).

    :param a1: First angle(s).
    :param a2: Second angle(s).
    :return: The signed distance(s).
    """
    if (isinstance(a1, np.ndarray) and a1.size == 0) or (isinstance(a2, np.ndarray) and a2.size == 0):
        return a1
    assert a1.shape == a2.shape or min(a1.shape) == 1 or min(a2.shape) == 1 or abs(len(a1.shape) - len(a2.shape)) == 1
    # diff = np.array(normalize_angle_symmetric(a2) - normalize_angle_symmetric(a1))
    diff = a2 - a1
    return normalize_angle_symmetric(diff)


def interpolate_poses(r1: float, p1: Union[float, np.ndarray], r2: float, p2: Union[float, np.ndarray]) -> \
        Tuple[float, float, float]:
    """Interpolate between the two SE2 poses (most importantly - correctly interpolate yaw across pi boundaries).

    :param r1: Ratio of the first pose. `r1 + r2` should equal to 1.
    :param p1: First pose (tuple x, y, yaw).
    :param r2: Ratio of the second pose. `r1 + r2` should equal to 1.
    :param p2: Second pose (tuple x, y, yaw).
    :return: The interpolated pose.
    """
    d_yaw = angular_signed_distance(p1[2], p2[2])
    p_x = r1 * p1[0] + r2 * p2[0]
    p_y = r1 * p1[1] + r2 * p2[1]
    p_yaw = normalize_angle_symmetric(p1[2] + d_yaw * r2)
    return p_x, p_y, p_yaw


def forward(u: np.ndarray, x0: Union[Sequence[float], np.ndarray]) -> np.ndarray:
    """Apply a sequence of control inputs to an initial state and return the whole trajectory.

    :param u: Control inputs (relative odometry). Numpy array 3xN.
    :param x0: The initial state. 3-tuple, Numpy array 3 or 3x1 or 1x3.
    :return: The trajectory in wcf. Numpy array 3x(N+1).
    """
    u = np.asarray(u, dtype=np.float64)
    x0 = np.asarray(x0, dtype=np.float64)
    pos_x, pos_y, phi = x0.ravel()
    traj = [col_vector((pos_x, pos_y, phi))]

    for i in range(u.shape[1]):
        traj.append(r2w(u[:, i], traj[-1]))
    return np.hstack(traj)


def r2w(z_r: Union[Sequence[float], np.ndarray], x: Union[Sequence[float], np.ndarray]) -> np.ndarray:
    """(rcf to wcf) Get wcf pose of the given measurements expressed in rcf (or apply relative odometry).

    :param z_r: The measurements or odometry in rcf. List or numpy array 3 or 3xN.
    :param x: Robot pose estimate in wcf in times corresponding to each measurement.
              List or numpy array 3 or 3xN or 3x(N+1) (last column ignored).
    :return: Wcf of the measurements computed using the robot pose estimates. Also expresses wcf poses in times x[t+1]
             if z_r are relative odometry measurements. Numpy array 3xN.
    """
    # convenience for passing lists
    x = np.asarray(x, dtype=np.float64)
    z_r = np.asarray(z_r, dtype=np.float64)

    # convenience to passing single measurements as 1D arrays
    return_1d = False
    if len(x.shape) == 1:
        x = x[:, np.newaxis]
        return_1d = True
    if len(z_r.shape) == 1:
        z_r = z_r[:, np.newaxis]

    # convenience for passing self.x which has one element more on the end
    if x.shape[1] == z_r.shape[1] + 1:
        x = x[:, :-1]

    # Construct the 2D rotation matrices
    sx, cx = np.sin(x[2, :]), np.cos(x[2, :])
    R = np.array([[cx, -sx], [sx, cx]])  # shape (2, 2, N)

    # Transform the measurements
    z_w = np.array(x)
    z_w[0:2, :] += np.einsum('ijk,jk->ik', R, z_r[0:2, :])
    z_w[2, :] = normalize_angle_symmetric(z_r[2, :] + x[2, :])  # Wrap yaw between -pi and pi

    # The above code does the same as the following, but more efficiently
    # z_w = np.zeros(x.shape)
    # for t in range(x.shape[1]):
    #     z_w[0:2, t] = R[:, :, t] @ z_r[0:2, t] + x[0:2, t]
    #     z_w[2, t] = normalize_angle_symmetric(z_r[2, t] + x[2, t])  # Wrap yaw between -pi and pi

    return z_w if not return_1d else z_w[:, 0]


def w2r(x2: Union[Sequence[float], np.ndarray], x: Union[Sequence[float], np.ndarray]) -> np.ndarray:
    """(wcf to rcf) Get rcf pose of the given wcf coordinates (or compute relative odometry).

    :param x2: The wcf coordinates to transform. List or numpy array 3 or 3xN.
    :param x: Robot pose estimate in wcf in times corresponding to each x2 coordinate.
              List or numpy array 3 or 3xN or 3x(N+1) (last column ignored).
    :return: Rcf of the x2 wcf coordinates computed using the robot pose estimates x. Also expresses relative odometry
             between x2[t] and x[t]. Numpy array 3xN.
    """
    # convenience for passing lists
    x = np.asarray(x, dtype=np.float64)
    x2 = np.asarray(x2, dtype=np.float64)

    return_1d = False
    # convenience to passing single coordinates as 1D arrays
    if len(x.shape) == 1:
        x = x[:, np.newaxis]
        return_1d = True
    if len(x2.shape) == 1:
        x2 = x2[:, np.newaxis]

    # convenience for passing self.x which has one element more on the end
    if x.shape[1] == x2.shape[1] + 1 and x2.shape[1] > 1:
        x = x[:, :-1]

    # Construct the 2D rotation matrices
    sx, cx = np.sin(x[2, :]), np.cos(x[2, :])
    R = np.array([[cx, sx], [-sx, cx]])  # shape (2, 2, N)
    if len(R.shape) == 2:
        R = R[::, np.newaxis]

    # Transform the coordinates
    z_r = np.zeros_like(x)
    np.einsum('ijk,jk->ik', R, x2[0:2, :] - x[0:2, :], out=z_r[0:2, :])
    z_r[2, :] = angular_signed_distance(x[2, :], x2[2, :])

    # The above code does the same as the following, but more efficiently
    # for t in range(x.shape[1]):
    #     x2_t = t if x2.shape[1] > 1 else 0
    #     z_r[0:2, t] = R[:, :, t] @ (x2[0:2, x2_t] - x[0:2, t])
    #     z_r[2, t] = angular_signed_distance(x[2, t], x2[2, x2_t])

    return z_r if not return_1d else z_r[:, 0]


def col_vector(x_list: Union[Sequence, np.ndarray]) -> np.ndarray:
    """
    Convert 1-dimensional Numpy array N to 2-dimensional numpy array Nx1.
    :param x_list: The 1D array. Numpy array N.
    :return: Numpy array Nx1.
    """
    x_col = np.array(x_list)[:, None]
    return x_col


def row_vector(x_list: Union[Sequence, np.ndarray]) -> np.ndarray:
    """
    Convert 1-dimensional Numpy array N to 2-dimensional numpy array 1xN.
    :param x_list: The 1D array. Numpy array N.
    :return: Numpy array 1xN.
    """
    x_row = np.array(x_list)[None, :]
    return x_row


def atleast_2d_row(x: np.ndarray) -> np.ndarray:
    """
    Like numpy.atleast_2d, but adds the new dimension as the second one instead of the first one.
    :param x: Input array (Numpy array N or Numpy array NxM).
    :return: The new array. Numpy array Nx1 or Numpy array NxM.
    """
    x = np.asarray(x)
    if len(x.shape) == 2:
        return x
    return x[:, np.newaxis]
