#!/usr/bin/env python3
"""
Simple path follower.

Always acts on the last received plan.
An empty plan means no action (stopping the robot).
"""

from __future__ import annotations

import random
from threading import RLock
from timeit import default_timer as timer

import actionlib
import numpy as np
import rospy
import tf2_ros
from aro_control.utils import (
    get_circ_line_intersect,
    line_point_dist,
    segment_point_dist,
    slots,
    tf_to_pose2d,
)
from geometry_msgs.msg import (
    Point,
    Pose,
    Pose2D,
    PoseStamped,
    Quaternion,
    Transform,
    TransformStamped,
    Twist,
)
from nav_msgs.msg import Path
from ros_numpy import msgify, numpify
from sensor_msgs.msg import PointCloud
from tf.transformations import euler_from_quaternion, quaternion_from_euler
from tf2_py import TransformException
from visualization_msgs.msg import Marker

from aro_msgs.msg import (
    FollowPathAction,
    FollowPathFeedback,
    FollowPathGoal,
    FollowPathResult,
)

np.set_printoptions(precision=3)


def normalize_angle_diff(target, current):
    """
    Calculate the shortest angular difference between two angles.

    :param target: target angle in radians
    :param current: current angle in radians
    :return: shortest angular difference (target - current) in radians
    """
    diff = target - current
    while diff > np.pi:
        diff -= 2 * np.pi
    while diff < -np.pi:
        diff += 2 * np.pi
    return diff


def closest_point_on_line_to_circle(
    line_point1,
    line_point2,
    circle_center,
    circle_radius,
):
    """
    Calculate the point on a line segment between two points that is closest to a circle.
    Rules:
    1. If line_point2 is inside the circle, return line_point2
    2. If the line intersects the circle, return the intersection point closer to line_point2
    3. Otherwise, return the closest point on the line to the circle

    Parameters
    ----------
    line_point1, line_point2: numpy arrays [x, y] defining the line segment
    circle_center: numpy array [x, y] for the center of the circle
    circle_radius: radius of the circle

    Returns
    -------
    closest_point: numpy array [x, y] of the closest point
    distance: distance from the circle to the line (negative if line intersects circle)

    """
    # Convert inputs to numpy arrays for vector operations
    p1 = np.array(line_point1, dtype=float)
    p2 = np.array(line_point2, dtype=float)
    center = np.array(circle_center, dtype=float)

    # Check if p2 is inside the circle
    p2_to_center_dist = np.linalg.norm(p2 - center)
    if p2_to_center_dist <= circle_radius:
        return p2, p2_to_center_dist - circle_radius

    # Vector from p1 to p2
    line_vector = p2 - p1

    # Unit vector in the direction of the line
    line_length = np.linalg.norm(line_vector)
    if line_length == 0:
        return p1, np.linalg.norm(p1 - center) - circle_radius
    line_direction = line_vector / line_length

    # Vector from p1 to circle center
    p1_to_center = center - p1

    # Project p1_to_center onto the line direction
    projection_length = np.dot(p1_to_center, line_direction)

    # Calculate the closest point on the infinite line
    closest_point_on_infinite_line = p1 + projection_length * line_direction

    # Find the closest point on the line segment
    if projection_length <= 0:
        closest_point = p1
    elif projection_length >= line_length:
        closest_point = p2
    else:
        closest_point = closest_point_on_infinite_line

    # Vector from circle center to closest point
    center_to_closest = closest_point - center

    # Distance from center to closest point
    distance_to_center = np.linalg.norm(center_to_closest)

    # Distance from circle to line (negative if line intersects circle)
    distance = distance_to_center - circle_radius

    # Check for intersections
    is_intersecting = False
    intersection_points = []

    # Only check for intersections if the line segment might intersect the circle
    if distance <= 0:
        # Calculate intersection points using the quadratic formula
        a = np.sum(line_vector**2)
        b = 2 * np.sum(line_vector * (p1 - center))
        c = np.sum((p1 - center) ** 2) - circle_radius**2

        discriminant = b**2 - 4 * a * c

        if discriminant >= 0:  # There are intersections with the infinite line
            t1 = (-b + np.sqrt(discriminant)) / (2 * a)
            t2 = (-b - np.sqrt(discriminant)) / (2 * a)

            # Check if intersection points are on the line segment (0 <= t <= 1)
            if 0 <= t1 <= 1:
                intersection_points.append((p1 + t1 * line_vector, t1))
            if 0 <= t2 <= 1:
                intersection_points.append((p1 + t2 * line_vector, t2))

            # If we have intersection points, sort them by distance to p2
            # and set the closest point to the intersection closer to p2
            if intersection_points:
                is_intersecting = True

                # Sort by parameter t (higher t means closer to p2)
                intersection_points.sort(key=lambda x: x[1], reverse=True)

                # Set the closest point to the intersection point closer to p2
                closest_point = intersection_points[0][0]

    return closest_point, distance


class PathFollower:
    def __init__(self):
        self.map_frame = rospy.get_param("~map_frame", "icp_map")
        self.odom_frame = rospy.get_param("~odom_frame", "odom")  # No-wait frame
        self.robot_frame = rospy.get_param(
            "~robot_frame",
            "base_footprint",
        )  # base_footprint for simulation
        self.control_freq = rospy.get_param(
            "~control_freq",
            2.0,
        )  # control loop frequency (Hz)
        assert 1.0 < self.control_freq <= 10.0
        # allowed distance from goal to be supposed reached
        self.goal_reached_dist = rospy.get_param("~goal_reached_dist", 0.2)
        self.final_orientation_tolerance = rospy.get_param(
            "~final_orientation_tolerance",
            0.1,
        )  # radians
        # maximum distance from a path start to enable start of path following
        self.max_path_dist = rospy.get_param("~max_path_dist", 4.0)
        self.max_velocity = rospy.get_param(
            "~max_velocity",
            1000,
        )  # maximum allowed velocity (m/s)
        self.max_angular_rate = rospy.get_param(
            "~max_angular_rate",
            1000,
        )  # maximum allowed angular rate (rad/s)
        self.look_ahead = rospy.get_param(
            "~look_ahead_dist",
            0.600,
        )  # look ahead distance for pure pursuit (m)
        # type of applied control_law approach PID/pure_pursuit
        self.control_law = rospy.get_param("~control_law", "pure_pursuit")
        assert self.control_law in ("PID", "pure_pursuit")

        self.lock = RLock()
        self.path_msg = None  # FollowPathGoal message
        self.path = None  # n-by-3 path array
        self.path_frame = None  # string
        self.path_index = 0  # Path position index
        self.min_point_dist = 0.5

        self.cmd_pub = rospy.Publisher(
            "cmd_vel",
            Twist,
            queue_size=1,
        )  # Publisher of velocity commands

        # Publisher of the lookahead point (visualization only)
        self.lookahead_point_pub = rospy.Publisher(
            "/vis/lookahead_point",
            Marker,
            queue_size=1,
        )
        self.path_pub = rospy.Publisher("/vis/path", Path, queue_size=2)
        self.waypoints_pub = rospy.Publisher("/vis/waypoints", PointCloud, queue_size=2)

        # initialize tf listener
        self.tf = tf2_ros.Buffer()
        self.tf_sub = tf2_ros.TransformListener(self.tf)

        # initialize action server
        self.action_server = actionlib.SimpleActionServer(
            "follow_path",
            FollowPathAction,
            execute_cb=self.control,
            auto_start=False,
        )
        self.action_server.register_preempt_callback(self.preempt_control)
        self.action_server.start()

        rospy.loginfo("Path follower initialized.")

    def lookup_transform(
        self,
        target_frame: str,
        source_frame: str,
        time: rospy.Time,
        no_wait_frame: str = None,
        timeout: rospy.Duration = rospy.Duration.from_sec(0.0),
    ) -> TransformStamped:
        """
        Returns transformation between two frames on specified time.

        :param target_frame: target frame (transform to)
        :param source_frame: source frame (transform from)
        :param time: reference time
        :param no_wait_frame: connection frame
        :param timeout: timeout for lookup transform

        :return: tf from source frame to target frame as TransformStamped
        """
        if no_wait_frame is None or no_wait_frame == target_frame:
            return self.tf.lookup_transform(
                target_frame,
                source_frame,
                time,
                timeout=timeout,
            )

        tf_n2t = self.tf.lookup_transform(self.map_frame, self.odom_frame, rospy.Time())
        tf_s2n = self.tf.lookup_transform(
            self.odom_frame,
            self.robot_frame,
            time,
            timeout=timeout,
        )
        tf_s2t = TransformStamped()
        tf_s2t.header.frame_id = target_frame
        tf_s2t.header.stamp = time
        tf_s2t.child_frame_id = source_frame
        tf_s2t.transform = msgify(
            Transform,
            np.matmul(numpify(tf_n2t.transform), numpify(tf_s2n.transform)),
        )
        return tf_s2t

    def get_robot_pose(self, target_frame: str) -> Pose2D:
        """
        Returns robot's pose in the specified frame.

        :param target_frame: the reference frame

        :return: current position of the robot in the reference frame
        """
        t = timer()
        tf = self.lookup_transform(
            target_frame,
            self.robot_frame,
            rospy.Time(),
            timeout=rospy.Duration.from_sec(0.5),
            no_wait_frame=self.odom_frame,
        )
        pose = tf_to_pose2d(tf.transform)
        return pose

    def clear_path(self):
        """
        Reinitializes path variables.
        """
        self.path_msg = None
        self.path = None
        self.path_index = 0

    def set_path_msg(self, msg: FollowPathGoal):
        """
        Initializes path variables based on received path request.

        :param msg containing header and required path (array of Pose2D)
        """
        with self.lock:
            if len(msg.poses) > 0:
                self.path_msg = msg
                self.path = np.array([slots(p) for p in msg.poses])
                self.path_index = 0  # path position index
                self.path_frame = msg.header.frame_id
            else:
                self.clear_path()

        self.path_received_time = rospy.Time.now()
        rospy.loginfo("Path received (%i poses).", len(msg.poses))

    def get_lookahead_point(self, pose: np.ndarray) -> np.ndarray | None:
        """
        Returns lookahead point used as a reference point for path following.

        :param pose: x, y coordinates and heading of the robot. Numpy 1D array 3x0

        :return: x, y coordinates of the lookahead point as Numpy 1D array 2x0 or None if the point cannot be found
        """
        # TODO: Find local goal (lookahead point) on current path
        assert self.path is not None

        p1i = min(self.path_index, self.path.shape[0] - 2)
        pd, dist = closest_point_on_line_to_circle(
            self.path[p1i][:2],
            self.path[p1i + 1][:2],
            pose[:2],
            self.look_ahead,
        )

        if dist > self.max_path_dist:
            return None

        return pd  # type:ignore

    def publish_path_visualization(self, path: FollowPathGoal):
        """
        Publishes a given path as sequence of lines and point cloud of particular waypoints.

        :param path: path to be visualized
        """
        if self.path_msg is None:
            return

        msg = PointCloud()
        msg.header.frame_id = self.map_frame
        msg.header.stamp = rospy.get_rostime()

        path_msg = Path()
        path_msg.header.frame_id = self.map_frame
        path_msg.header.stamp = rospy.get_rostime()

        for p in path.poses:
            msg.points.append(Point(x=p.x, y=p.y, z=0.0))
            pose_stamped = PoseStamped()
            pose_stamped.pose.position = msg.points[-1]
            path_msg.poses.append(pose_stamped)

        self.waypoints_pub.publish(msg)
        self.path_pub.publish(path_msg)

    def publish_lookahead_point(self, lookahead_point_pose: np.ndarray):
        """
        Publishes a given pose as a red cicular marker.

        :param lookahead_point_pose: desired x, y coordinates of the marker in map frame. Numpy 1D array 2x0
        """
        msg = Marker()
        msg.header.frame_id = self.map_frame
        msg.header.stamp = rospy.get_rostime()
        msg.id = 1
        msg.type = 2
        msg.action = 0
        msg.pose = Pose()
        msg.pose.orientation = Quaternion(0.0, 0.0, 0.0, 1.0)
        msg.pose.position = Point(lookahead_point_pose[0], lookahead_point_pose[1], 0.0)
        msg.color.r = 1.0
        msg.color.g = 0.0
        msg.color.b = 0.0
        msg.color.a = 1.0
        msg.scale.x = 0.05
        msg.scale.y = 0.05
        msg.scale.z = 0.01
        self.lookahead_point_pub.publish(msg)

    def send_velocity_command(self, linear_velocity: float, angular_rate: float):
        """
        Calls command to set robot velocity and angular rate.

        :param linear_velocity: desired forward linear velocity
        :param angular_rate: desired angular rate
        """
        msg = Twist()
        msg.angular.z = angular_rate
        msg.linear.x = linear_velocity
        self.cmd_pub.publish(msg)

    def control(self, msg: FollowPathGoal):
        """
        Callback function of action server. Starts the path following process.

        :param msg: msg containing header and required path (array of Pose2D)
        """
        rospy.loginfo("New control request obtained.")
        self.set_path_msg(msg)
        rate = rospy.Rate(self.control_freq)
        pose = np.array([0.0, 0.0, 0.0])

        self.publish_path_visualization(self.path_msg)

        if self.path_msg is None:
            rospy.logwarn("Empty path msg received.")
            self.send_velocity_command(0, 0)
            # TODO: stop the robot if empty path is received

            if self.action_server.is_active():
                self.action_server.set_succeeded(
                    FollowPathResult(Pose2D(pose[0], pose[1], 0)),
                    text="Goal reached.",
                )
            return

        # Main control loop.
        while True:
            t = timer()  # save current time

            rospy.loginfo_throttle(1.0, "Following path.")
            pose = np.array([0.0, 0.0, 0.0])
            with self.lock:
                # get robot pose
                try:
                    if self.path_frame is None:
                        rospy.logwarn(
                            "No valid path received so far, returning zero position.",
                        )
                    else:
                        if self.path_msg is None:
                            pose_msg = self.get_robot_pose(self.path_frame)
                        else:
                            pose_msg = self.get_robot_pose(
                                self.path_msg.header.frame_id,
                            )

                        pose = np.array(slots(pose_msg))

                except TransformException as ex:
                    rospy.logerr("Robot pose lookup failed: %s.", ex)
                    self.send_velocity_command(0.0, 0.0)
                    continue

                if self.path_msg is None:
                    rospy.logwarn(
                        "Path following was preempted. Leaving the control loop.",
                    )
                    self.send_velocity_command(0.0, 0.0)
                    return

                # get lookahead point
                lookahead_point = self.get_lookahead_point(pose)
                last = len(self.path_msg.poses) - 1

            if lookahead_point is None:
                self.send_velocity_command(0.0, 0.0)
                if self.action_server.is_active():
                    self.action_server.set_aborted(
                        FollowPathResult(Pose2D(pose[0], pose[1], 0)),
                        text="Distance too high.",
                    )
                return

            # publish visualization of lookahead point
            self.publish_lookahead_point(lookahead_point)

            # Position displacement (direction and Euclidean distance)
            lookahead_point_dir = lookahead_point - pose[:2]
            lookahead_point_dist = np.linalg.norm(lookahead_point_dir)

            k_point = 0.8
            if self.path_index < last and (
                np.linalg.norm(self.path[self.path_index + 1][:2] - pose[:2])
                < k_point * self.min_point_dist
            ):
                self.path_index += 1

            goal_dist = np.linalg.norm(self.path[last][:2] - pose[:2])
            goal_reached = (
                self.path_index == last and self.goal_reached_dist >= goal_dist
            )
            goal_orientation_diff = abs(
                normalize_angle_diff(self.path[last][2], pose[2]),
            )
            orientation_reached = (
                self.final_orientation_tolerance >= goal_orientation_diff
            )

            # Clear path and stop if the goal has been reached.
            if goal_reached:
                rospy.loginfo(
                    "Goal reached: %.2f m from robot (<= %.2f m) in %.2f s.",
                    lookahead_point_dist,
                    self.goal_reached_dist,
                    (rospy.Time.now() - self.path_received_time).to_sec(),
                )

                # TODO: stop the robot when it reached the goal

                if orientation_reached:
                    with self.lock:
                        if self.action_server.is_active():
                            self.action_server.set_succeeded(
                                FollowPathResult(Pose2D(pose[0], pose[1], 0)),
                                text="Goal reached.",
                            )
                        self.send_velocity_command(0.0, 0.0)
                        self.clear_path()
                        # self.spin()
                    return
                target_orientation = self.path[last][2]
                orientation_error = normalize_angle_diff(target_orientation, pose[2])

                # Apply proportional control to orientation
                k_orient = 1.0  # Proportional gain for orientation control
                angular_rate = k_orient * orientation_error

                # Apply limits on angular rate
                angular_rate = np.clip(
                    angular_rate,
                    -self.max_angular_rate,
                    self.max_angular_rate,
                )

                # Stop linear motion and only rotate
                self.send_velocity_command(0.0, angular_rate)

                self.action_server.publish_feedback(
                    FollowPathFeedback(
                        Pose2D(pose[0], pose[1], pose[2]),
                        0.0,
                        angular_rate,
                        0.0,
                    ),
                )
                rate.sleep()
                continue

            lookahead_point_too_far = lookahead_point_dist > self.max_path_dist

            # Clear path and stop if the path is too far.
            if lookahead_point_too_far:
                rospy.logwarn(
                    "Distance to path %.2f m too high (> %.2f m).",
                    lookahead_point_dist,
                    self.max_path_dist,
                )

                # TODO: stop the robot if it is too far from the path

                with self.lock:
                    if self.action_server.is_active():
                        self.action_server.set_aborted(
                            FollowPathResult(Pose2D(pose[0], pose[1], 0)),
                            text="Distance too high.",
                        )
                    self.send_velocity_command(0.0, 0.0)
                    self.clear_path()
                return

            # TODO: apply control law to produce control inputs
            th = pose[2]
            dir_vec = np.array([np.cos(th), np.sin(th)])
            dpv = lookahead_point_dir - np.dot(dir_vec, lookahead_point_dir) * dir_vec
            dp = np.linalg.norm(dpv)
            r = lookahead_point_dist**2 / (2 * dp)

            base_velocity = self.max_velocity

            # k_curv = 2.0
            # curvature_factor = 1.0 / (1.0 + k_curv * abs(1 / r))

            steering_weight = 0.8
            steering_angle = np.arcsin(dp / lookahead_point_dist)
            max_steering_angle = np.pi * 0.45
            min_steering_angle = np.pi / 15
            dir_s = np.sign(np.dot(lookahead_point_dir, dir_vec))
            steering_factor = (
                1.0 - (abs(steering_angle) / max_steering_angle) * steering_weight
                if steering_angle > min_steering_angle or dir_s < 0
                else 1.0
            )

            goal_deceleration_distance = self.goal_reached_dist * 2
            if self.path_index == last:
                goal_factor = min(1.0, goal_dist / goal_deceleration_distance)
            else:
                goal_factor = 1.0

            velocity = base_velocity * goal_factor * steering_factor * dir_s
            velocity = np.clip(velocity, 0.0, self.max_velocity)
            s = np.sign(np.dot([-lookahead_point_dir[1], lookahead_point_dir[0]], dpv))
            if np.allclose(velocity, 0) or steering_angle > min_steering_angle:
                angular_rate = dir_s * s * self.max_angular_rate
            else:
                angular_rate = s * velocity / r

            # apply limits on angular rate and linear velocity
            angular_rate = np.clip(
                angular_rate,
                -self.max_angular_rate,
                self.max_angular_rate,
            )

            # Apply desired velocity and angular rate
            self.send_velocity_command(velocity, angular_rate)

            self.action_server.publish_feedback(
                FollowPathFeedback(
                    Pose2D(pose[0], pose[1], 0),
                    velocity,
                    angular_rate,
                    lookahead_point_dist,
                ),
            )
            rospy.logdebug(
                "Speed: %.2f m/s, angular rate: %.1f rad/s. (%.3f s), "
                "pose = [%.2f, %.2f], lookahead_point = [%.2f, %.2f], time = %.2f",
                velocity,
                angular_rate,
                timer() - t,
                pose[0],
                pose[1],
                lookahead_point[0],
                lookahead_point[1],
                rospy.get_rostime().to_sec(),
            )
            rate.sleep()

    def spin(self):
        self.send_velocity_command(0, self.max_angular_rate)
        rospy.sleep((2 * np.pi) / self.max_angular_rate)
        self.send_velocity_command(0, 0)

    def preempt_control(self):
        """
        Preemption callback function of action server. Safely preempts the path following process.
        """
        with self.lock:
            # TODO: implement reaction on request to preempt the control (stop the robot and discard a path being followed)
            self.send_velocity_command(0, 0)
            pose = np.array([0.0, 0.0, 0.0])  # TODO: replace by your code
            self.path = None
            self.path_msg = None

            if self.action_server.is_active():
                self.action_server.set_aborted(
                    FollowPathResult(Pose2D(pose[0], pose[1], 0)),
                    text="Control preempted.",
                )

            rospy.logwarn("Control preempted.")


if __name__ == "__main__":
    rospy.init_node("path_follower", log_level=rospy.INFO)
    node = PathFollower()
    rospy.spin()
