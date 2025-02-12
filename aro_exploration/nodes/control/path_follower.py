#!/usr/bin/env python3
"""
Simple path follower.

Always acts on the last received plan.
An empty plan means no action (stopping the robot).
"""

from __future__ import absolute_import, division, print_function
import rospy
import numpy as np
from ros_numpy import msgify, numpify
from tf.transformations import euler_from_quaternion, quaternion_from_euler
from tf2_py import TransformException
import tf2_ros
from threading import RLock
from timeit import default_timer as timer
import actionlib
from aro_msgs.msg import FollowPathAction, FollowPathFeedback, FollowPathResult, FollowPathGoal
from visualization_msgs.msg import Marker
from nav_msgs.msg import Path
from sensor_msgs.msg import PointCloud
from geometry_msgs.msg import Pose, Pose2D, Quaternion, Transform, TransformStamped, Twist, PoseStamped, Point
from aro_control.utils import segment_point_dist, get_circ_line_intersect, line_point_dist, slots, tf_to_pose2d
import random

np.set_printoptions(precision=3)

class PathFollower(object):
    def __init__(self):
        self.map_frame = rospy.get_param('~map_frame', 'icp_map')
        self.odom_frame = rospy.get_param('~odom_frame', 'odom')  # No-wait frame
        self.robot_frame = rospy.get_param('~robot_frame', 'base_footprint')  # base_footprint for simulation
        self.control_freq = rospy.get_param('~control_freq', 2.0)  # control loop frequency (Hz)
        assert 1.0 < self.control_freq <= 10.0
        # allowed distance from goal to be supposed reached
        self.goal_reached_dist = rospy.get_param('~goal_reached_dist', 0.2)
        # maximum distance from a path start to enable start of path following
        self.max_path_dist = rospy.get_param('~max_path_dist', 4.0)
        self.max_velocity = rospy.get_param('~max_velocity', 0.20)  # maximum allowed velocity (m/s)
        self.max_angular_rate = rospy.get_param('~max_angular_rate', 0.5)  # maximum allowed angular rate (rad/s)
        self.look_ahead = rospy.get_param('~look_ahead_dist', 1.00)  # look ahead distance for pure pursuit (m)
        # type of applied control_law approach PID/pure_pursuit
        self.control_law = rospy.get_param('~control_law', "pure_pursuit")
        assert self.control_law in ('PID', 'pure_pursuit')

        self.lock = RLock()
        self.path_msg = None  # FollowPathGoal message
        self.path = None  # n-by-3 path array
        self.path_frame = None  # string
        self.path_index = 0  # Path position index

        self.cmd_pub = rospy.Publisher('cmd_vel', Twist, queue_size=1)  # Publisher of velocity commands

        # Publisher of the lookahead point (visualization only)
        self.lookahead_point_pub = rospy.Publisher('/vis/lookahead_point', Marker, queue_size=1)
        self.path_pub = rospy.Publisher('/vis/path', Path, queue_size=2)
        self.waypoints_pub = rospy.Publisher('/vis/waypoints', PointCloud, queue_size=2)

        # initialize tf listener
        self.tf = tf2_ros.Buffer()
        self.tf_sub = tf2_ros.TransformListener(self.tf)

        # initialize action server
        self.action_server = actionlib.SimpleActionServer(
            'follow_path', FollowPathAction, execute_cb=self.control, auto_start=False)
        self.action_server.register_preempt_callback(self.preempt_control)
        self.action_server.start()

        rospy.loginfo('Path follower initialized.')

    def lookup_transform(self, target_frame: str, source_frame: str, time: rospy.Time,
                         no_wait_frame: str = None,
                         timeout: rospy.Duration = rospy.Duration.from_sec(0.0)) -> TransformStamped:
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
            return self.tf.lookup_transform(target_frame, source_frame, time, timeout=timeout)

        tf_n2t = self.tf.lookup_transform(self.map_frame, self.odom_frame, rospy.Time())
        tf_s2n = self.tf.lookup_transform(self.odom_frame, self.robot_frame, time, timeout=timeout)
        tf_s2t = TransformStamped()
        tf_s2t.header.frame_id = target_frame
        tf_s2t.header.stamp = time
        tf_s2t.child_frame_id = source_frame
        tf_s2t.transform = msgify(Transform,
                                  np.matmul(numpify(tf_n2t.transform), numpify(tf_s2n.transform)))
        return tf_s2t

    def get_robot_pose(self, target_frame: str) -> Pose2D:
        """
        Returns robot's pose in the specified frame.

        :param target_frame: the reference frame

        :return: current position of the robot in the reference frame
        """
        t = timer()
        tf = self.lookup_transform(target_frame, self.robot_frame, rospy.Time(),
                                   timeout=rospy.Duration.from_sec(0.5), no_wait_frame=self.odom_frame)
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
                self.path_index = 0 # path position index
                self.path_frame = msg.header.frame_id
            else:
                self.clear_path()

        self.path_received_time = rospy.Time.now()
        rospy.loginfo('Path received (%i poses).', len(msg.poses))

    def get_lookahead_point(self, pose: np.ndarray) -> np.ndarray:
        """
        Returns lookahead point used as a reference point for path following.

        :param pose: x, y coordinates and heading of the robot. Numpy 1D array 3x0

        :return: x, y coordinates of the lookahead point as Numpy 1D array 2x0 or None if the point cannot be found
        """

        # TODO: Find local goal (lookahead point) on current path

        local_goal = self.path[self.path_index][:2]  # TODO: replace by your code

        return local_goal

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
        rospy.loginfo('New control request obtained.')
        self.set_path_msg(msg)
        rate = rospy.Rate(self.control_freq)
        pose = np.array([0.0, 0.0, 0.0])

        self.publish_path_visualization(self.path_msg)

        if self.path_msg is None:
            rospy.logwarn('Empty path msg received.')
            # TODO: stop the robot if empty path is received

            if self.action_server.is_active():
                self.action_server.set_succeeded(FollowPathResult(Pose2D(pose[0], pose[1], 0)), text='Goal reached.')
            return

        # Main control loop.
        while True:
            t = timer()  # save current time

            rospy.loginfo_throttle(1.0, 'Following path.')
            pose = np.array([0.0, 0.0, 0.0])
            with self.lock:
                # get robot pose
                try:
                    if self.path_frame is None:
                        rospy.logwarn('No valid path received so far, returning zero position.')
                    else:
                        if self.path_msg is None:
                            pose_msg = self.get_robot_pose(self.path_frame)
                        else:
                            pose_msg = self.get_robot_pose(self.path_msg.header.frame_id)

                        pose = np.array(slots(pose_msg))

                except TransformException as ex:
                    rospy.logerr('Robot pose lookup failed: %s.', ex)
                    self.send_velocity_command(0.0, 0.0)
                    continue

                if self.path_msg is None:
                    rospy.logwarn('Path following was preempted. Leaving the control loop.')
                    self.send_velocity_command(0.0, 0.0)
                    return

                # get lookahead point
                lookahead_point = self.get_lookahead_point(pose)
                last = len(self.path_msg.poses) - 1

            if lookahead_point is None:
                self.send_velocity_command(0.0, 0.0)
                if self.action_server.is_active():
                    self.action_server.set_aborted(FollowPathResult(Pose2D(pose[0], pose[1], 0)), text='Distance too high.')
                return

            # publish visualization of lookahead point
            self.publish_lookahead_point(lookahead_point)

            # Position displacement (direction and Euclidean distance)
            lookahead_point_dir = lookahead_point - pose[:2]
            lookahead_point_dist = np.linalg.norm(lookahead_point_dir)

            goal_reached = False  # TODO: replace by your code

            # Clear path and stop if the goal has been reached.
            if goal_reached:
                rospy.loginfo('Goal reached: %.2f m from robot (<= %.2f m) in %.2f s.',
                              lookahead_point_dist, self.goal_reached_dist, (rospy.Time.now() - self.path_received_time).to_sec())

                # TODO: stop the robot when it reached the goal


                with self.lock:
                    if self.action_server.is_active():
                        self.action_server.set_succeeded(FollowPathResult(Pose2D(pose[0], pose[1], 0)), text='Goal reached.')
                    self.send_velocity_command(0.0, 0.0)
                    self.clear_path()
                return

            lookahead_point_too_far = False  # TODO: replace by your code

            # Clear path and stop if the path is too far.
            if lookahead_point_too_far:
                rospy.logwarn('Distance to path %.2f m too high (> %.2f m).', lookahead_point_dist, self.max_path_dist)

                # TODO: stop the robot if it is too far from the path


                with self.lock:
                    if self.action_server.is_active():
                        self.action_server.set_aborted(FollowPathResult(Pose2D(pose[0], pose[1], 0)), text='Distance too high.')
                    self.send_velocity_command(0.0, 0.0)
                    self.clear_path()
                return

            # TODO: apply control law to produce control inputs
            angular_rate = -0.5 + random.random()  # TODO: replace by your code
            velocity = 0.1 * random.random()  # TODO: replace by your code


            # apply limits on angular rate and linear velocity
            angular_rate = np.clip(angular_rate, -self.max_angular_rate, self.max_angular_rate)
            velocity = np.clip(velocity, 0.0, self.max_velocity)

            # Apply desired velocity and angular rate
            self.send_velocity_command(velocity, angular_rate)

            self.action_server.publish_feedback(FollowPathFeedback(Pose2D(pose[0], pose[1], 0), velocity, angular_rate, lookahead_point_dist))
            rospy.logdebug('Speed: %.2f m/s, angular rate: %.1f rad/s. (%.3f s), '
                           'pose = [%.2f, %.2f], lookahead_point = [%.2f, %.2f], time = %.2f',
                           velocity, angular_rate, timer() - t,
                           pose[0], pose[1], lookahead_point[0], lookahead_point[1], rospy.get_rostime().to_sec())
            rate.sleep()

    def preempt_control(self):
        """
        Preemption callback function of action server. Safely preempts the path following process.
        """
        with self.lock:

            # TODO: implement reaction on request to preempt the control (stop the robot and discard a path being followed)
            pose = np.array([0.0, 0.0, 0.0]) # TODO: replace by your code

            if self.action_server.is_active():
                self.action_server.set_aborted(FollowPathResult(Pose2D(pose[0], pose[1], 0)), text='Control preempted.')

            rospy.logwarn('Control preempted.')

if __name__ == '__main__':
    rospy.init_node('path_follower', log_level=rospy.INFO)
    node = PathFollower()
    rospy.spin()
