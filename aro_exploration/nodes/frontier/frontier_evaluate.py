#!/usr/bin/env python3
from __future__ import absolute_import, division, print_function

# Setup python paths first.
import os
import sys
import rospy
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import TransformStamped
import subprocess
from threading import Timer
import tf.transformations as tft
import tf2_ros
from aro_msgs.srv import GenerateFrontier
from timeit import default_timer as c_timer

dir = os.path.dirname(os.path.abspath(__file__))

print('Python version: %s' % sys.version)
print('Python version info: %s' % (sys.version_info,))

import base64
from glob import glob
import mimetypes
import numpy as np
from shutil import copytree

#np.set_printoptions(suppress=True)

def check_if_pos_is_numbers(coords):
    return (coords and coords.goal_pose and (coords is not None) and (coords.goal_pose.x is not None) and (not np.isnan(
        coords.goal_pose.x)))

class FrontierTester():
    def __init__(self):
        self.gridSubscriber = rospy.Subscriber('occupancy', OccupancyGrid, self.grid_cb)
        self.tf_pub = tf2_ros.TransformBroadcaster()
        self.x = 0
        self.y = 0

    def shutdown(self):
        self.gridSubscriber.unregister()

    def grid_cb(self, msg):
        self.publish_position()


    def publish_position(self):
        tf_msg = TransformStamped()
        tf_msg.header.stamp = rospy.Time.now()
        tf_msg.transform.translation.x = self.x
        tf_msg.transform.translation.y = self.y
        tf_msg.transform.translation.z = 0
        q = tft.quaternion_from_euler(0, 0, 0)
        tf_msg.transform.rotation.x = q[0]
        tf_msg.transform.rotation.y = q[1]
        tf_msg.transform.rotation.z = q[2]
        tf_msg.transform.rotation.w = q[3]
        tf_msg.header.frame_id = 'icp_map'
        tf_msg.child_frame_id = 'base_footprint'
        self.tf_pub.sendTransform(tf_msg)

        tf_msg = TransformStamped()
        tf_msg.header.stamp = rospy.Time.now()
        tf_msg.header.frame_id = 'map'
        tf_msg.child_frame_id = 'icp_map'
        tf_msg.transform.rotation.w = 1
        self.tf_pub.sendTransform(tf_msg)

    def set_pos(self, x, y):
        self.x = x
        self.y = y


if __name__ == '__main__':
    total = 0.0
    messages = ''
    max_dist = 0.3 # Acceptable imprecision.
    output = []
    endsleeptime = 5
    sleepbetweentrials = 1

    if not os.path.isfile(os.path.join(dir,'frontier.py')):
        rospy.loginfo('Frontier script not found.')
        pass
    else:
        roscore_proc = subprocess.Popen(['roscore'])
        rospy.init_node("tester")
        # rospy.set_param('robot_diameter', 0.8)
        # rospy.set_param('occupancy_treshold', 90)
        # rospy.set_param('map_frame', 'icp_map')
        # rospy.set_param('robot_frame', 'base_footprint')
        player_proc = subprocess.Popen(['rosbag', 'play', '../../data/planning/map_dat.bag'], cwd=dir)

        rviz_proc = subprocess.Popen(['roslaunch', 'aro_exploration', 'aro_frontier_planning_rviz.launch'])

        kill = lambda process : process.terminate()

        script_output_file = 'frontier_output.txt'
        with open(script_output_file, 'w') as f_output:
            frontier_proc = subprocess.Popen(['rosrun', 'aro_exploration', 'frontier.py'], stdout=f_output, stderr=f_output)

        rospy.sleep(5)
        timeout = 30
        scan_timer = Timer(timeout, kill, [frontier_proc])
        scan_timer2 = Timer(timeout, kill, [roscore_proc])
        scan_timer3 = Timer(timeout, kill, [player_proc])
        serv_start = False

        frontier = FrontierTester()
        frontier.set_pos(0,0)

        try:
            scan_timer.start()
            scan_timer2.start()
            scan_timer3.start()
            t = c_timer()
            try:
                rospy.wait_for_service('get_closest_frontier', 10)
                get_frontier = rospy.ServiceProxy('get_closest_frontier', GenerateFrontier)
                serv_start = True
            except Exception as e: 
                rospy.loginfo(f'[EVALUATION] Exception caught: {e}.') 

            if serv_start:
                score = 0
                test_positions = [
                    (-0.3, 0.7, -0.8, 1.0),
                    (-0.3, -1.0, -1.05, -0.9),
                    (-0.2, -1.7, -0.65, -1.95),
                    (0.6, 1.7, 0.45, 1.75),
                    # (1.5, -2.0, 1.9, -1.95)
                    (-0.6, 0.3, -0.775, 1.025) # this example has starting position near obstacle
                ]

                for i, (px, py, fx, fy) in enumerate(test_positions, start=1):
                    frontier.set_pos(px, py)
                    rospy.sleep(sleepbetweentrials)
                    rospy.loginfo(f'Test {i}:')
                    try:
                        coords = get_frontier()
                        if coords and coords.goal_pose:
                            if check_if_pos_is_numbers(coords):
                                x, y = coords.goal_pose.x, coords.goal_pose.y
                                if np.sqrt((fx - x) ** 2 + (fy - y) ** 2) > max_dist:
                                    rospy.loginfo(f'Incorrect solution - Frontier center [{x}, {y}] received when robot is at [{px}, {py}] is too far from reference solution [{fx}, {fy}]')
                                else:
                                    rospy.loginfo(f'SUCCESS! Frontier center [{x}, {y}] received when robot is at [{px}, {py}] is sufficiently close to reference solution [{fx}, {fy}]')
                                    score += 1
                            else:
                                rospy.logwarn(f'Incorrect solution - received frontier center coords are nan. ')

                    except Exception as e:
                        rospy.loginfo(f'[EVALUATION] Exception caught: {e}.')
                        if (c_timer() - t) < timeout:
                            rospy.loginfo("Exception caught during getting frontier. Check script's output.")
                        else:
                            rospy.loginfo("Internal timeout exceeded. Get closest frontier service response time is too high.")               

                rospy.loginfo("[EVALUATION] Total evaluation success rate: %.2f / %.2f", score, len(test_positions)) 


        except Exception as e:
            rospy.loginfo(f'[EVALUATION] Exception caught: {e}.')

        if not serv_start:
            rospy.loginfo('Timeout while waiting for service to start.')

        rospy.loginfo("[EVALUATION] Your node's console output has been written to " + script_output_file + " in your current directory.")
        rospy.loginfo("[EVALUATION] Waiting for %.2f seconds and exiting", endsleeptime) 
        rospy.loginfo("[EVALUATION] You can change the sleep time between trials and at the end in this script") 
        rospy.sleep(endsleeptime)
        
        scan_timer.cancel()
        scan_timer2.cancel()
        scan_timer3.cancel()

        #frontier.shutdown()
        frontier_proc.terminate()
        player_proc.terminate()
        rviz_proc.terminate()
        roscore_proc.terminate()
