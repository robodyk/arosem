#!/usr/bin/env python3
"""
Simple exploration control node navigating the robot to the closest frontier.
A random frontier or random poses can be used for recovery.
"""

from aro_msgs.srv import GenerateFrontier, PlanPath, PlanPathRequest, PlanPathResponse
from geometry_msgs.msg import Pose, Pose2D, PoseStamped, PoseArray, Quaternion, Transform, TransformStamped
from aro_msgs.msg import FollowPathAction, FollowPathFeedback, FollowPathResult, FollowPathGoal
from aro_msgs.msg import Path
from std_srvs.srv import SetBool, SetBoolRequest
import actionlib
import numpy as np
from ros_numpy import msgify, numpify
import rospy
from tf.transformations import euler_from_quaternion, quaternion_from_euler
from tf2_py import TransformException
import tf2_ros
from threading import RLock
from typing import List, Optional, Tuple

np.set_printoptions(precision=3)


def slots(msg):
    """Return message attributes (slots) as list."""
    return [getattr(msg, var) for var in msg.__slots__]


def array(msg):
    """Return message attributes (slots) as array."""
    return np.array(slots(msg))


def pose2to3(pose2: Pose2D) -> Pose:
    """Convert Pose2D to Pose."""
    pose3 = Pose()
    pose3.position.x = pose2.x
    pose3.position.y = pose2.y
    rpy = 0.0, 0.0, pose2.theta
    q = quaternion_from_euler(*rpy)
    pose3.orientation = Quaternion(*q)
    return pose3


def tf3to2(tf: Transform) -> Pose2D:
    """Convert Transform to Pose2D."""
    pose2 = Pose2D()
    pose2.x = tf.translation.x
    pose2.y = tf.translation.y
    rpy = euler_from_quaternion(slots(tf.rotation))
    pose2.theta = rpy[2]
    return pose2


def pose_valid(pose: Pose2D):
    return ~np.isnan(slots(pose)).any()


class Explorer(object):
    def __init__(self):
        rospy.loginfo('Initializing explorer.')
        self.map_frame = rospy.get_param('~map_frame', 'map')
        self.odom_frame = rospy.get_param('~odom_frame', 'odom')
        self.robot_frame = rospy.get_param('~robot_frame', 'base_footprint')
        self.goal_reached_dist = rospy.get_param('~goal_reached_dist', 0.2)
        self.retries = int(rospy.get_param('~retries', 3))
        self.run_mode = rospy.get_param('~run_mode', 'manual')
        assert self.retries >= 1

        self.lock = RLock()
        self.marker_pose: Optional[Pose] = None
        self.current_goal: Optional[Pose2D] = None
        self.current_path: Optional[List[Pose2D]] = None
        self.frontiers_left = False

        rospy.loginfo('Initializing services and publishers.')

        # Using frontier.py
        self.get_closest_frontier = rospy.ServiceProxy('get_closest_frontier', GenerateFrontier)
        self.wait_for_service(self.get_closest_frontier)
        self.get_best_value_frontier = rospy.ServiceProxy('get_best_value_frontier', GenerateFrontier)
        self.wait_for_service(self.get_best_value_frontier)

        if self.run_mode == 'eval':
            self.stop_sim = rospy.ServiceProxy('stop_simulation', SetBool)
            self.wait_for_service(self.stop_sim)



        self.tf = tf2_ros.Buffer()
        self.tf_sub = tf2_ros.TransformListener(self.tf)

        self.plan_path = rospy.ServiceProxy('plan_path', PlanPath)
        self.wait_for_service(self.plan_path)

        # For exploration / path_follower
        self.path_pub = rospy.Publisher('path', Path, queue_size=2)
        self.poses_pub = rospy.Publisher('poses', PoseArray, queue_size=2)
        self.marker_sub = rospy.Subscriber('relative_marker_pose', PoseStamped, self.relative_marker_pose_cb)
        self.follow_path = actionlib.SimpleActionClient('follow_path', FollowPathAction)
        while not self.follow_path.wait_for_server(timeout=rospy.Duration(1.0)) and not rospy.is_shutdown():
            rospy.logwarn("Waiting for follow_path action server")

        self.tf = tf2_ros.Buffer()
        self.tf_sub = tf2_ros.TransformListener(self.tf)

        # Get initial pose
        self.init_pose: Optional[Pose2D] = None
        while self.init_pose is None and not rospy.is_shutdown():
            try:
                self.init_pose = self.get_robot_pose()
                rospy.loginfo('Explorer initial pose:')
                rospy.loginfo(self.init_pose)
            except:
                rospy.loginfo('Robot odometry is not ready yet.')
                self.init_pose = None
                rospy.sleep(1.0)

        rospy.loginfo('Initializing timer.')
        self.timer = rospy.Timer(rospy.Duration(1.0), self.update_plan)

    def wait_for_service(self, service: rospy.ServiceProxy):
        while not rospy.is_shutdown():
            try:
                service.wait_for_service(timeout=1.0)
                rospy.loginfo("Service %s is available now.", service.resolved_name)
                return
            except rospy.ROSInterruptException:
                return
            except rospy.ROSException:
                rospy.logwarn("Waiting for service %s" % (service.resolved_name,))

    def lookup_transform(self, target_frame: str, source_frame: str, time: rospy.Time,
                         no_wait_frame: str = None, timeout=rospy.Duration.from_sec(0.0)) -> TransformStamped:
        if no_wait_frame is None or no_wait_frame == target_frame:
            return self.tf.lookup_transform(target_frame, source_frame, time, timeout=timeout)

        tf_n2t = self.tf.lookup_transform(self.map_frame, self.odom_frame, rospy.Time())
        tf_s2n = self.tf.lookup_transform(self.odom_frame, self.robot_frame, time, timeout=timeout)
        tf_s2t = TransformStamped()
        tf_s2t.header.frame_id = target_frame
        tf_s2t.header.stamp = time
        tf_s2t.child_frame_id = source_frame
        tf_s2t.transform = msgify(Transform,
                                  np.matmul(numpify(tf_n2t.transform),
                                            numpify(tf_s2n.transform)))
        return tf_s2t

    def relative_marker_pose_cb(self, msg: PoseStamped):
        self.marker_pose = msg.pose
        rospy.loginfo('Marker localized at pose: [%.1f,%.1f].', self.marker_pose.position.x,
                      self.marker_pose.position.y)

    def get_robot_pose(self) -> Pose2D:
        tf = self.lookup_transform(self.map_frame, self.robot_frame, rospy.Time.now(),
                                   timeout=rospy.Duration.from_sec(0.5), no_wait_frame=self.odom_frame)
        pose = tf3to2(tf.transform)
        return pose

    def get_feasible_goal(self) -> Tuple[Optional[Pose2D], Optional[List[Pose2D]]]:
        rospy.loginfo('Looking for goal.')

        # TODO implement the exploration logic
        goal: Optional[Pose2D] = None
        path: Optional[List[Pose2D]] = None


        return goal, path

    def update_plan(self, event):
        try:
            with self.lock:
                pose = array(self.get_robot_pose())
                while self.current_goal is not None:
                    rospy.loginfo_throttle(3, 'Following path to to (%.2f, %.2f).', self.current_goal.x, self.current_goal.y)

                # TODO check whether the marker was found and robot returned to starting pose
                finished = False


                # Terminate evaluation instance
                if self.run_mode == 'eval' and finished:
                    self.timer.shutdown()
                    req = SetBoolRequest(True)
                    self.stop_sim(req)

                # Find new goal
                self.current_goal, self.current_path = self.get_feasible_goal()
                if self.current_goal is None:
                    rospy.logwarn('Feasible goal not found during %i retries.',
                                  self.retries)
                    return
                if self.current_path is None:
                    rospy.logwarn('Feasible goal found, but no path. Weird.')
                    return

                # Delegate to path follower.
                follow_path_goal = FollowPathGoal()
                follow_path_goal.header.stamp = rospy.Time.now()
                follow_path_goal.header.frame_id = self.map_frame
                follow_path_goal.poses = self.current_path

                self.follow_path.send_goal(follow_path_goal, feedback_cb=self.follow_path_feedback_cb,
                                           done_cb=self.follow_path_done_cb)

                poses = PoseArray()
                poses.header = follow_path_goal.header
                poses.poses = [pose2to3(p) for p in self.current_path]
                self.poses_pub.publish(poses)

                # self.last_planned = rospy.Time.now()
                rospy.loginfo('Goal updated to (%.2f, %.2f).', self.current_goal.x, self.current_goal.y)
        except TransformException as ex:
            rospy.logerr('Robot pose lookup failed: %s.', ex)

    def follow_path_feedback_cb(self, feedback: FollowPathFeedback):
        rospy.loginfo('Received control feedback, tracking error is: %.2f.', feedback.error)

    def follow_path_done_cb(self, state, result: FollowPathResult):
        rospy.loginfo('Control done.')
        self.current_goal = None


if __name__ == '__main__':
    rospy.init_node('explorer', log_level=rospy.INFO)
    node = Explorer()
    rospy.spin()
