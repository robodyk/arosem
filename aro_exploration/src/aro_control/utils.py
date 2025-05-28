import sys
from typing import Any, List

import numpy as np
from geometry_msgs.msg import Pose, Pose2D, Quaternion, Transform
from tf.transformations import euler_from_quaternion, quaternion_from_euler

try:
    import rospy

    logwarn = rospy.logwarn
    logerr = rospy.logerr
except ImportError:
    logwarn = print
    logerr = print


def segment_point_dist(
    seg_start: np.ndarray,
    seg_end: np.ndarray,
    point: np.ndarray,
) -> float:
    """
    Returns distance of point from a segment.

    :param seg_start: x, y coordinates of segment beginning. Numpy 1D array 2x0.
    :param seg_end: x, y coordinates of segment end. Numpy 1D array 2x0.
    :param point: x, y coordinates of point. Numpy 1D array 2x0.

    :return: Euclidean distance between segment and point as float.
    """
    seg = seg_end - seg_start
    len_seg = np.linalg.norm(seg)

    if len_seg == 0:
        return np.linalg.norm(seg_start - point)

    t = max(0.0, min(1.0, np.dot(point - seg_start, seg) / len_seg**2))
    proj = seg_start + t * seg

    return np.linalg.norm(point - proj)


def point_slope_form(p1: np.ndarray, p2: np.ndarray) -> (float, float, float):
    """
    Returns coefficients of a point-slope form of a line equation given by two points.

    :param p1: x, y coordinates of first point. Numpy 1D array 2x0.
    :param p2: x, y coordinates of second point. Numpy 1D array 2x0.

    :return: a, b, c (float): coefficients of a point-slope form of a line equation ax + by + c = 0.
    """
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]

    if dx != 0:
        return dy / dx, -1, -(dy / dx) * p1[0] + p1[1]
    return 1, 0, -p1[0]


def get_circ_line_intersect(
    p1: np.ndarray,
    p2: np.ndarray,
    circ_c: np.ndarray,
    circ_r: float,
) -> List[np.ndarray]:
    """
    Returns intersection points of a line given by two points
    and circle given by its center and radius.

    :param p1: x, y coordinates of point on the line. Numpy 1D array 2x0.
    :param p2: x, y coordinates of second point on the line. Numpy 1D array 2x0.
    :param circ_c: x, y coordinates of circle center. Numpy 1D array 2x0.
    :param circ_r: circle radius

    :return:  x, y coordinates of intersection points, list of Numpy 1D array 2x0.
    """
    a, b, c = point_slope_form(p1, p2)  # get point-slope form of line ax + by + c = 0

    # find intersection based on line and circle equation
    if b != 0:  # line is not parallel to y axis
        const_t = (
            circ_c[0] ** 2
            + 2 * circ_c[1] * c / b
            + circ_c[1] ** 2
            + c**2 / b**2
            - circ_r**2
        )
        lin_t = 2 * a * c / b**2 + 2 * circ_c[1] * a / b - 2 * circ_c[0]
        quad_t = 1 + a**2 / b**2
        x_vals = np.roots([quad_t, lin_t, const_t])  # find roots of quadratic equation
        y_vals = [
            -a / b * x - c / b for x in x_vals
        ]  # compute y from substitution in line eq.
    else:
        const_t = c**2 + 2 * circ_c[0] * c + circ_c[0] ** 2 + circ_c[1] ** 2 - circ_r**2
        lin_t = -2 * circ_c[1]
        quad_t = 1
        y_vals = np.real(np.roots([quad_t, lin_t, const_t]))
        x_vals = [p1[0] for i in y_vals]  # compute x from substitution in line eq.

    points = [[x_vals[i], y_vals[i]] for i in range(len(x_vals))]  # intersection points
    return points


def line_point_dist(
    line_point_1: np.ndarray,
    line_point_2: np.ndarray,
    point: np.ndarray,
) -> float:
    """
    Returns distance of point from a line.

    :param line_point_1: x, y coordinates of first point on the line. Numpy 1D array 2x0.
    :param line_point_2: x, y coordinates of second point on the line. Numpy 1D array 2x0.
    :param point: x, y coordinates of a point. Numpy 1D array 2x0.

    :return: Euclidean distance between line and point (float).
    """
    p = point - line_point_1
    v = line_point_2 - line_point_1
    return abs((v[0]) * (p[1]) - (p[0]) * (v[1])) / np.linalg.norm(v[:2])


def slots(msg: Any) -> List[Any]:
    """
    Returns message attributes (slots) as list.

    :param msg: input ROS msg.

    :return: list of attributes of message (list).
    """
    return [getattr(msg, var) for var in msg.__slots__]


def tf_to_pose2d(tf: Transform) -> Pose2D:
    """
    Converts tf to Pose2D.

    :param tf: tf to be converted.

    :return: tf converted to geometry_msgs/Pose2D.
    """
    pose2 = Pose2D()
    pose2.x = tf.translation.x
    pose2.y = tf.translation.y
    rpy = euler_from_quaternion(slots(tf.rotation))
    pose2.theta = rpy[2]
    return pose2


def array(msg: Any) -> np.ndarray:
    """
    Converts message attributes (slots) to array.

    :param tf: tf to be converted.

    :return: message  attributes as Numpy 1D array.
    """
    return np.array(slots(msg))


def pose2to3(pose2: Pose2D) -> Pose:
    """
    Converts Pose2D to Pose.

    :param pose2: Pose2D to be converted.

    :return: pose2 converted to geometry_msgs/Pose.
    """
    pose3 = Pose()
    pose3.position.x = pose2.x
    pose3.position.y = pose2.y
    rpy = 0.0, 0.0, pose2.theta
    q = quaternion_from_euler(*rpy)
    pose3.orientation = Quaternion(*q)
    return pose3


def pose2_to_array(pose: Pose2D) -> np.ndarray:
    """
    Converts Pose2D to array.

    :param pose: Pose2D to be converted.

    :return: pose2 converted to np.ndarray.
    """
    return np.array([pose.x, pose.y])


def path_point_dist(path: List[Pose2D], point: np.ndarray) -> float:
    """
    Returns distance of point from a path.

    :param path: x, y, theta coordinates of path waypoints. List of Pose2D.
    :param point: x, y, theta coordinates of point. Numpy 1D array 3x0.

    :return: Euclidean distance between path and point as float.
    """
    if path is None or len(path) < 2:
        logwarn(
            "Cannot compute dist to path for empty path or path containing a single point.",
        )
        return -1.0

    min_dist = sys.float_info.max
    for k in range(len(path) - 1):
        dist = segment_point_dist(
            pose2_to_array(path[k]),
            pose2_to_array(path[k + 1]),
            point[:2],
        )
        min_dist = min(min_dist, dist)

    if min_dist > 0.3:
        logerr("Error, distance from path too high")

    return min_dist
