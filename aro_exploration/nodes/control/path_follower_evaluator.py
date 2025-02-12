#!/usr/bin/env python3
"""
Simple exploration control node navigating the robot to the closest frontier.
A random frontier or random poses can be used for recovery.
"""
from __future__ import absolute_import, division, print_function
from geometry_msgs.msg import Pose2D, PoseStamped, PoseArray, Transform, TransformStamped
from nav_msgs.msg import Path
from sensor_msgs.msg import PointCloud
import numpy as np
from ros_numpy import msgify, numpify
import rospy
from tf2_py import TransformException
import tf2_ros
from threading import RLock
import actionlib
from aro_msgs.msg import FollowPathAction, FollowPathFeedback, FollowPathResult, FollowPathGoal
from aro_control.utils import array, slots, path_point_dist, pose2to3, tf_to_pose2d
from std_msgs.msg import Float32
import csv
import time
from timeit import default_timer as timer
from prettytable import PrettyTable

np.set_printoptions(precision=3)

class EvaluatorAction(object):
    def __init__(self):
        self.map_frame = rospy.get_param('~map_frame', 'map')
        self.odom_frame = rospy.get_param('~odom_frame', 'odom')
        self.robot_frame = rospy.get_param('~robot_frame', 'base_footprint')
        self.goal_reached_dist = rospy.get_param('~goal_reached_dist', 0.2)
        self.paths_filename = rospy.get_param('~paths_filename', 'test_paths.csv')
        self.results_filename = rospy.get_param('~results_filename', 'results.csv')
        self.save_results = rospy.get_param('~save_results', False)
        self.max_path_deviation = rospy.get_param('~max_path_deviation', 0.2)
        self.lock = RLock()
        self.current_pose = None
        self.current_goal = None  # Pose2D
        self.current_path = None  # Pose2D[]
        self.current_following_time_start = rospy.Time.now()
        self.current_following_time_end = rospy.Time.now()
        self.current_following_dev = []  # floats in meters
        self.overall_following_dev = []  # tuples (avg, min, max) in m
        self.overall_following_times = []  # floats in seconds
        self.goals_reached = []  # bools
        self.following_time_limits = []
        self.max_start_dist = 0.0

        # For exploration / path_follower
        self._ac = actionlib.SimpleActionClient('follow_path', FollowPathAction)

        self.path_pub = rospy.Publisher('/vis/path_eval', Path, queue_size=2)
        self.waypoints_pub = rospy.Publisher('/vis/waypoints_eval', PointCloud, queue_size=2)

        self.deviation_pub = rospy.Publisher('/vis/path_deviation', Float32, queue_size=1)
        self.vel_pub = rospy.Publisher('/vis/applied_vel', Float32, queue_size=1)
        self.ang_rate_pub = rospy.Publisher('/vis/applied_ang_rate', Float32, queue_size=1)

        self.tf = tf2_ros.Buffer()
        self.tf_sub = tf2_ros.TransformListener(self.tf)

        rospy.loginfo('Waiting for path follower action server')
        self._ac.wait_for_server()
        rospy.loginfo('Action server found')

        time.sleep(5) # TODO wait for slam and tfs

        rospy.loginfo('Start evaluation process')
        self.run_evaluation(self.paths_filename)
        rospy.loginfo('Evaluation process finished.')

    def publish_visualizations(self, path):
        msg = PointCloud()
        msg.header.frame_id = self.map_frame
        msg.header.stamp = rospy.get_rostime()
        for p in path.poses:
            msg.points.append(p.pose.position)

        self.waypoints_pub.publish(msg)
        self.path_pub.publish(path)

    def lookup_transform(self, target_frame, source_frame, time,
                         no_wait_frame=None, timeout=rospy.Duration.from_sec(0.0)):
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

    def get_robot_pose(self):
        tf = self.lookup_transform(self.map_frame, self.robot_frame, rospy.Time.now(),
                                   timeout=rospy.Duration.from_sec(0.5), no_wait_frame=self.odom_frame)
        pose = tf_to_pose2d(tf.transform)
        return pose

    def load_paths(self, filename):
        path_list = []
        with open(filename, 'r') as f:
            path = []
            r = csv.reader(f)
            for row in r:

                if 'PATH_END' in row:
                    rospy.loginfo('Path end detected.')
                    self.following_time_limits.append(float(row[1]))
                    path_list.append(path)
                    path = []
                    continue

                point = [float(v) for v in row]
                path.append(Pose2D(point[0], point[1], 0.0))

        rospy.loginfo('Loaded %d paths.', len(path_list))
        return path_list

    def process_and_save_data(self, filename, num_paths, empty_path_result): 
        
        if self.save_results:
            rospy.loginfo('Saving evaluation data to %s.', filename)
            f = open(filename, "w")
            line = "goal_reached, dev_ok, time_ok, follow_time, time_diff, avg_dev, min_dev, max_dev\n"
            f.write(line)

        results_table = PrettyTable(['Path idx', 'goal reached', 'dev ok', 'time ok', 'follow time', 'time diff', 'avg dev', 'min dev', 'max dev'])

        overall_dev_ok = True
        overall_time_ok = True
        all_paths_followed = True
        score = 0.0
        for k in range(0, len(self.goals_reached)):
            time_diff = self.overall_following_times[k] - self.following_time_limits[k]
            dev_ok = self.overall_following_dev[k][2] - self.max_path_deviation < 0
            time_ok = time_diff < 0
            overall_dev_ok = overall_dev_ok and dev_ok
            overall_time_ok = overall_time_ok and time_ok
            score += 4.5/float(num_paths) * self.goals_reached[k] * (0.5 * dev_ok + 0.5 * time_ok)
            
            results_table.add_row([str(k + 1), str(self.goals_reached[k]), str(dev_ok), str(time_ok), "{:.3f}".format(self.overall_following_times[k]), "{:.3f}".format(time_diff), "{:.3f}".format(self.overall_following_dev[k][0]), "{:.3f}".format(self.overall_following_dev[k][1]), "{:.3f}".format(self.overall_following_dev[k][2])])

            if self.save_results:
                line = ("Path " + str(k + 1) + ": " + str(self.goals_reached[k]) + ", " + str(dev_ok) + ", " +
                        str(time_ok) + ", " + "{:.3f}".format(self.overall_following_times[k]) + ", " +
                        "{:.3f}".format(time_diff) + ", " + "{:.3f}".format(self.overall_following_dev[k][0]) +
                        ", " + "{:.3f}".format(self.overall_following_dev[k][1]) + ", " +
                        "{:.3f}".format(self.overall_following_dev[k][2]) + "\n")
                f.write(line)

        if not len(self.goals_reached) == num_paths:
            all_paths_followed = False

        score += 0.5 * empty_path_result

        results_table.align['follow time'] = "r"
        results_table.align['time diff'] = "r"
        print("----------------------------------------------\n")
        print("RESULTS:")
        print(results_table)

        lines = []
        # lines.append("\n")
        lines.append("All paths followed: OK\n" if all_paths_followed else (
                "All paths followed: failed (" + str(len(self.goals_reached)) + "/" + str(num_paths)+ " followed)\n"))
        lines.append("Deviation limits: OK\n" if overall_dev_ok else "Deviation limits: failed\n")
        lines.append("Time limits: OK\n" if overall_time_ok else "Time limits: failed\n")
        lines.append("Stop on empty path: OK\n" if empty_path_result else "Stop on empty path: failed\n")

        if overall_time_ok and overall_dev_ok and empty_path_result and all_paths_followed:
            lines.append("Overall status: OK\n")
        elif score >= 3.0: 
            lines.append("Overall status: acceptable\n")
        else:
            lines.append("Overall status: failed\n")

        lines.append("Score: " + str(round(score, 1)))

        for line in lines:
            if self.save_results:
                f.write(line)
            print(line)

        if self.save_results:
            f.close()
            rospy.loginfo('Evaluation data saved.')

    def run_evaluation(self, filename):
        paths = self.load_paths(filename)

        self.overall_following_dev = []  # tuples (avg, min, max) in m
        self.overall_following_times = []  # floats in seconds
        self.goals_reached = []  # bools
        timeout_reached = False
        empty_path_result = False

        t = timer()

        for path in paths:
            self.current_following_time_start = rospy.Time.now() 
            self.current_following_dev = [] # floats in meters
            self.perform_single_plan(path)
            self.current_goal = path[-1] 
            self.max_start_dist = 0.0

            # wait for following start
            time.sleep(6.0)

            while self.current_goal is not None:
                rospy.loginfo('Waiting for end of path following.')
                time.sleep(1.0)
                if timer() - t > 360.0:
                    timeout_reached = True
                    rospy.loginfo('Time limit for path evaluation exceeded.')
                    break

            if timeout_reached:
                self._ac.cancel_all_goals()
                break

            # process data
            dist = 0.0
            goal = self.current_path[-1]

            if self.max_start_dist < self.goal_reached_dist:
                rospy.logwarn('Path following has not started at all.')

            try:
                with self.lock:
                    pose = array(self.get_robot_pose())
                    dist = np.sqrt((pose[0] - goal.x)**2 + (pose[1] - goal.y)**2)
            except TransformException as ex:
                rospy.logerr('Robot pose lookup failed: %s.', ex)

            self.goals_reached.append(dist <= self.goal_reached_dist < self.max_start_dist)
            self.overall_following_times.append(
                (self.current_following_time_end - self.current_following_time_start).to_sec())

            if len(self.current_following_dev):
                avg_dev = np.mean(self.current_following_dev)
                min_dev = min(self.current_following_dev)
                max_dev = max(self.current_following_dev)
            else:
                avg_dev = -1
                min_dev = -1
                max_dev = -1

            self.overall_following_dev.append((avg_dev, min_dev, max_dev))

        if not timeout_reached:
            empty_path_result = self.test_empty_path(paths[0])

        self.process_and_save_data(self.results_filename, len(paths), empty_path_result)

    def test_empty_path(self, path):

        t = timer()

        self.max_start_dist = 0.0
        self.perform_single_plan(path)
        self.current_goal = path[-1]

        # wait for following start
        time.sleep(8.0)

        pose_start = Pose2D()
        pose_end = Pose2D()

        empty_path = []
        self.perform_single_plan(empty_path)
        start_time = rospy.Time.now()
        timeout_reached = False

        try:
            with self.lock:
                pose_start = array(self.get_robot_pose())
        except TransformException as ex:
            rospy.logerr('Robot pose lookup failed: %s.', ex)

        while self.current_goal is not None:
            rospy.loginfo_throttle(1.0, 'Waiting for end of path following.')
            time.sleep(0.1)
            if timer() - t > 20.0:
                timeout_reached = True
                self._ac.cancel_all_goals()
                rospy.loginfo('Time limit for empty path evaluation exceeded.')
                break

        following_started = True
        if self.max_start_dist < self.goal_reached_dist:
            rospy.logwarn('Path following has not started at all.')
            following_started = False

        stop_time = rospy.Time.now()

        try:
            with self.lock:
                pose_end = array(self.get_robot_pose())
        except TransformException as ex:
            rospy.logerr('Robot pose lookup failed: %s.', ex)

        # compute distance travelled from time when the empty path command was sent
        dist = np.sqrt((pose_start[0] - pose_end[0])**2 + (pose_start[1] - pose_end[1])**2)

        rospy.loginfo('Empty path result: following_started = %d, timeout_reached = %d, dist = %.2f, stop_time = %.2f.',
                      following_started, timeout_reached, dist, (stop_time - start_time).to_sec())
        return following_started and not timeout_reached and dist < 2*self.max_path_deviation and \
            (stop_time - start_time).to_sec() < 2.0

    def perform_single_plan(self, plan):
        # check that goal is reached when action server returns done, check that robot stops if the path is empty list
        # check that new path is followed once it is published
        try:
            with self.lock:
                self._ac.cancel_all_goals()
                self.current_path = plan
                path_stamped = Path()
                path_stamped.header.stamp = rospy.Time.now()
                path_stamped.header.frame_id = self.map_frame
                path_stamped.poses = [PoseStamped(pose=pose2to3(p)) for p in self.current_path]
                time.sleep(1)
                self._ac.send_goal(FollowPathGoal(path_stamped.header, self.current_path),
                                   feedback_cb=self.action_feedback_cb, done_cb=self.action_done_cb)

                poses = PoseArray()
                poses.header = path_stamped.header
                poses.poses = [pose2to3(p) for p in self.current_path]
                self.publish_visualizations(path_stamped)

                rospy.loginfo('New path sent to path follower.')

        except TransformException as ex:
            rospy.logerr('Robot pose lookup failed: %s.', ex)

    def action_feedback_cb(self, feedback):
        # get robot pose and its distance from current path
        try:
            with self.lock:
                pose = array(self.get_robot_pose())

            dist = path_point_dist(self.current_path, pose)
            if dist < 0:
                rospy.logerr('Distance to path cannot be computed since path is too short.')

            if len(self.current_path) > 0:
                self.max_start_dist = max(
                    self.max_start_dist,
                    np.sqrt((pose[0] - self.current_path[0].x)**2 + (pose[1] - self.current_path[0].y)**2))

            self.current_following_dev.append(dist)

        except TransformException as ex:
            rospy.logerr('Robot pose lookup failed: %s.', ex)

        self.deviation_pub.publish(Float32(dist))
        self.vel_pub.publish(Float32(feedback.linear_velocity))
        self.ang_rate_pub.publish(Float32(feedback.angular_rate))

        rospy.loginfo_throttle(2.0, 'Received control feedback. Position = [%.2f, %.2f], applied control: vel.= %.2f m/s, ang. r.= %.2f rad/s, lookahead point dist = %.2f m.',
                               feedback.position.x, feedback.position.y, feedback.linear_velocity, feedback.angular_rate, feedback.error)

    def action_done_cb(self, state, result):
        self.current_following_time_end = rospy.Time.now()
        self.current_goal = None
        rospy.loginfo('Control done. %s Final position = [%.2f, %.2f]',
                      self._ac.get_goal_status_text(), result.finalPosition.x, result.finalPosition.y)


if __name__ == '__main__':
    rospy.init_node('explorer', log_level=rospy.INFO)
    node = EvaluatorAction()
    rospy.spin()
