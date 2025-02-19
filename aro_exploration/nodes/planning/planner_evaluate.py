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
from aro_msgs.srv import PlanPath, PlanPathRequest, PlanPathResponse
from geometry_msgs.msg import Pose2D, Pose, PoseStamped, Point, Quaternion
from scipy.ndimage import morphology

dir = os.path.dirname(os.path.abspath(__file__))

import base64
from glob import glob
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import mimetypes
import numpy as np
from shutil import copytree

np.set_printoptions(suppress=True)

from nav_msgs.msg import MapMetaData
from aro_exploration.utils import get_circular_dilation_footprint


def get_path_total_length(path, is_path_object = True):
    total_len = 0

    for i in range(len(path) - 1):
        increment = 0
        if is_path_object:
            increment = np.sqrt((path[i].x - path[i + 1].x) ** 2 + ((path[i].y - path[i + 1].y) ** 2))
        else:
            increment = np.sqrt((path[i][0] - path[i + 1][0]) ** 2 + ((path[i][1] - path[i + 1][1]) ** 2))

        total_len += increment
    return total_len

def check_path_increments_length(path, max_path_increment_length, is_path_object = True):
    big_inc = False
    big_inc_pos1 = None
    big_inc_pos2 = None

    for i in range(len(path) - 1):
        increment = 0
        if is_path_object:
            increment = np.sqrt((path[i].x - path[i + 1].x) ** 2 + ((path[i].y - path[i + 1].y) ** 2))
        else:
            increment = np.sqrt((path[i][0] - path[i + 1][0]) ** 2 + ((path[i][1] - path[i + 1][1]) ** 2))

        if increment > max_path_increment_length:
            big_inc = True
            big_inc_pos1 = path[i]
            big_inc_pos2 = path[i+1]

    return big_inc, big_inc_pos1, big_inc_pos2

def check_if_pos_safe(x, y, evaluator):
    robot_diameter = evaluator.robot_diameter
    grid_resolution = evaluator.grid_resolution
    dilation_footprint = get_circular_dilation_footprint(robot_diameter, grid_resolution)

    dilation_footprint_width = dilation_footprint.shape[0]
    dilation_radius_cells = (int)((dilation_footprint.shape[0] - 1) / 2)

    translated_pos_hom = np.array([x - evaluator.gridInfo.origin.position.x, y - evaluator.gridInfo.origin.position.y, 0, 1])
    quat = evaluator.gridInfo.origin.orientation
    mat = tft.quaternion_matrix(tft.quaternion_inverse([quat.x, quat.y, quat.z, quat.w]))
    grid_pos_unrounded = (mat.dot(translated_pos_hom[np.newaxis].T).flatten()[:2]) / evaluator.gridInfo.resolution
    roundedPos = np.round(grid_pos_unrounded)
    grid_pos = roundedPos if np.allclose(grid_pos_unrounded, roundedPos) else np.floor(grid_pos_unrounded)
    grid_x, grid_y = grid_pos

    reason = ""
    safe = True

    if grid_x < dilation_radius_cells or grid_x >= evaluator.width - dilation_radius_cells or grid_y < dilation_radius_cells or grid_y >= evaluator.height - dilation_radius_cells:
        reason = "Pos too close to edges of grid"
        safe = False
        return safe, reason

    dilation_where = np.transpose(np.where(dilation_footprint))
    delta_coords = dilation_where - np.array([1, 1]).reshape((1,2)) * dilation_radius_cells

    for dc in delta_coords:
        xx = (int)(grid_x + dc[0])
        yy = (int)(grid_y + dc[1])
        val = evaluator.occupancy[yy, xx]
        if val < 0:
            return False,  "Pos " + str(np.array([x,y])) + " is unsafe! Nearby cell at grid_x: " + str(xx) + ", grid_y: " + str(yy) + " is UNKNOWN"
        if val > evaluator.occupancy_threshold:
            return False,  "Pos " + str(np.array([x,y])) + " is unsafe! Nearby cell at grid_x: " + str(xx) + ", grid_y: " + str(yy) + " is OCCUPIED"

    # Is ok
    return True, ""
        

def check_path(evaluator, gx, gy, messages, path, ref_path, sx, sy):
    success = False

    min_len = 0.1
    max_len_mod = 1.5
    max_path_increment_length = 0.2
    start_and_end_max_dist_from_path = 0.2
    clearing_dist = 1.2 * evaluator.robot_diameter / 2

    if path and path[0] and path is not None:
        total_len = 0
        man_len = np.abs(gx - sx) + np.abs(gy - sy)
        big_inc = False
        inval_poses = []

        # CHECK IF LENGTH OK 
        total_len = get_path_total_length(path)
        ref_len = get_path_total_length(ref_path, is_path_object = False)
        rospy.loginfo("Evaluated path length: " + str(total_len))
        rospy.loginfo("Reference solution length: " + str(ref_len))

        # CHECK IF NO LARGE JUMP IN PATH
        big_inc, big_inc_pos1, big_inc_pos2 = check_path_increments_length(path, max_path_increment_length)

        # CHECK SAFETY
        path_safe = True
        unsafety_reason = ""
        for i in range(len(path)):
            # ONLY CHECK OUTSIDE OF CLEARING DIST!
            if np.sqrt((path[i].x - sx) ** 2 + ((path[i].y - sy) ** 2)) > clearing_dist:
                pos_safe, reason = check_if_pos_safe(path[i].x, path[i].y, evaluator)
                if not pos_safe:
                    path_safe = False
                    unsafety_reason = reason
                    break

        # CHECK IF PATH CONNECTED TO START AND GOAL POSITIONS
        start_pos_dist = np.sqrt((path[0].x - sx) ** 2 + ((path[0].y - sy) ** 2))
        goal_pos_dist = np.sqrt((path[-1].x - gx) ** 2 + ((path[-1].y - gy) ** 2))

        # EVAL AND WRITE ERRORS
        if total_len < min_len:
            messages += 'Path length from [' + str(sx) + ', ' + str(sy) + '] to [' + str(gx) + ', ' + str(
                gy) + '] length too short.</p>'
        elif total_len >= ref_len * max_len_mod:
            messages += 'Path length from [' + str(sx) + ', ' + str(sy) + '] to [' + str(gx) + ', ' + str(
                gy) + '] length too long. Path length cannot be higher than ' + str(int(max_len_mod * 100)) + '% of reference solution </p>'
        elif big_inc:
            messages += 'Path length from [' + str(sx) + ', ' + str(sy) + '] to [' + str(gx) + ', ' + str(
                gy) + '] has too big jump between ' + str(np.array([big_inc_pos1.x, big_inc_pos1.y])) + ' and ' + str(
                    np.array([big_inc_pos2.x, big_inc_pos2.y])) + '. Max allowed path increment length: ' + str(max_path_increment_length) + ' meters .</p>' 
        elif start_pos_dist > start_and_end_max_dist_from_path:
            messages += 'Path from [' + str(sx) + ', ' + str(sy) + '] to [' + str(gx) + ', ' + str(
                gy) + '] incorrect - start position distance from first path waypoint too big: ' + str(
                    start_pos_dist) + ', max dist: ' + str(start_and_end_max_dist_from_path) + '</p>' 
        elif goal_pos_dist > start_and_end_max_dist_from_path:
            messages += 'Path from [' + str(sx) + ', ' + str(sy) + '] to [' + str(gx) + ', ' + str(
                gy) + '] incorrect - goal position distance from last path waypoint too big: ' + str(
                    goal_pos_dist) + ', max dist: ' + str(start_and_end_max_dist_from_path) + '</p>' 
        elif not path_safe:
            messages += 'Path from [' + str(sx) + ', ' + str(sy) + '] to [' + str(gx) + ', ' + str(
                gy) + '] is unsafe because '+ unsafety_reason + ' .</p>'
            success = False
        else:
            messages += 'Path from [' + str(sx) + ', ' + str(sy) + '] to [' + str(gx) + ', ' + str(
                gy) + '] is valid.</p>'
            success = True

    else:
        messages += 'Path format is invalid.</p>'
    return messages, success

class PlanningEvaluator():
    def __init__(self):
        self.gridSubscriber = rospy.Subscriber('occupancy', OccupancyGrid, self.grid_cb)
        self.robot_diameter = float(rospy.get_param("~robot_diameter", 0.6))
        self.occupancy_threshold = int(rospy.get_param("~occupancy_threshold", 25))

        self.tf_pub = tf2_ros.TransformBroadcaster()
        self.x = 0
        self.y = 0
        self.robotDiameter = 0.6
        self.grid_resolution = 0.05
        self.gridReady = False

    def shutdown(self):
        self.gridSubscriber.unregister()

    def grid_cb(self, msg):
        if not self.gridReady:
            self.width, self.height, self.grid_resolution = msg.info.width, msg.info.height, msg.info.resolution
            self.gridInfo = msg.info
            self.occupancy = np.reshape(msg.data, (self.height, self.width))
            self.gridReady = True

        # Publish the currently cached robot position as static transform
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

def main():

    endsleeptime = 5
    sleepbetweentrials = 1

    total = 0.0
    messages = ''
    output = []
    max_dist = 0.3

    roscore_proc = subprocess.Popen(['roscore'])
    rospy.init_node("tester")
    rospy.set_param('robot_diameter', 0.6)
    rospy.set_param('occupancy_treshold', 90)
    rospy.set_param('map_frame', 'map')
    player_proc = subprocess.Popen(['rosbag', 'play', '../../data/planning/map_dat.bag'], cwd=dir)

    rviz_proc = subprocess.Popen(['roslaunch', 'aro_exploration', 'aro_frontier_planning_rviz.launch'])

    evaluator = PlanningEvaluator()
    kill = lambda process : process.terminate()

    planner_script_output_file = 'planner_output.txt'
    with open(planner_script_output_file, 'w') as f_output:
        planner_proc = subprocess.Popen(['rosrun', 'aro_exploration', 'planner.py'], stdout=f_output, stderr=f_output)

    rospy.sleep(5)
    timeout = 30
    scan_timer1 = Timer(timeout, kill, [planner_proc])
    scan_timer2 = Timer(timeout, kill, [roscore_proc])
    scan_timer3 = Timer(timeout, kill, [player_proc])
    serv_start = False

    scan_timer1.start()
    scan_timer2.start()
    scan_timer3.start()

    # Init reference paths
    reference_paths_map = []
    start_positions = []
    goal_positions = []

    # PATH 1
    # start_positions.append((-0.75, 1))
    start_positions.append((-0.75, 1))
    goal_positions.append((0.9, 1.8))
    reference_paths_map.append([(-0.72, 1.03), (-0.67, 1.03), (-0.62, 1.03), (-0.57, 1.03), (-0.52, 1.03), (-0.47, 1.03), (-0.42, 1.08), (-0.37, 1.13), (-0.32, 1.13), (-0.27, 1.13), (-0.22, 1.13), (-0.17, 1.13), (-0.12, 1.13), (-0.07, 1.13), (-0.02, 1.13), (0.03, 1.13), (0.08, 1.13), (0.13, 1.13), (0.18, 1.18), (0.23, 1.18), (0.28, 1.18), (0.33, 1.23), (0.38, 1.28), (0.43, 1.33), (0.48, 1.38), (0.53, 1.43), (0.58, 1.48), (0.63, 1.53), (0.68, 1.58), (0.73, 1.63), (0.78, 1.68), (0.83, 1.73), (0.88, 1.78), (0.93, 1.83)])

    # PATH 2
    start_positions.append((1.1, -0.8))
    goal_positions.append((0.9, 1.8))
    reference_paths_map.append([(1.13, -0.77), (1.08, -0.77), (1.03, -0.77), (0.98, -0.72), (0.93, -0.67), (0.88, -0.67), (0.83, -0.62), (0.78, -0.62), (0.73, -0.62), (0.68, -0.57), (0.63, -0.52), (0.58, -0.47), (0.53, -0.42), (0.53, -0.37), (0.53, -0.32), (0.48, -0.27), (0.48, -0.22), (0.48, -0.17), (0.48, -0.12), (0.48, -0.07), (0.48, -0.02), (0.48, 0.03), (0.48, 0.08), (0.48, 0.13), (0.48, 0.18), (0.48, 0.23), (0.48, 0.28), (0.48, 0.33), (0.48, 0.38), (0.48, 0.43), (0.48, 0.48), (0.48, 0.53), (0.48, 0.58), (0.48, 0.63), (0.48, 0.68), (0.48, 0.73), (0.48, 0.78), (0.48, 0.83), (0.48, 0.88), (0.48, 0.93), (0.48, 0.98), (0.48, 1.03), (0.48, 1.08), (0.48, 1.13), (0.48, 1.18), (0.48, 1.23), (0.53, 1.28), (0.58, 1.33), (0.63, 1.38), (0.68, 1.43), (0.73, 1.48), (0.78, 1.53), (0.83, 1.58), (0.83, 1.63), (0.88, 1.68), (0.88, 1.73), (0.88, 1.78), (0.93, 1.83)])

    # PATH 3
    start_positions.append((0.9, 1.8))
    goal_positions.append((-1.0, -0.9))
    reference_paths_map.append([(0.93, 1.83), (0.93, 1.78), (0.88, 1.73), (0.88, 1.68), (0.83, 1.63), (0.78, 1.58), (0.78, 1.53), (0.73, 1.48), (0.68, 1.43), (0.63, 1.38), (0.63, 1.33), (0.63, 1.28), (0.58, 1.23), (0.53, 1.18), (0.48, 1.13), (0.43, 1.08), (0.43, 1.03), (0.38, 0.98), (0.38, 0.93), (0.38, 0.88), (0.33, 0.83), (0.28, 0.78), (0.23, 0.73), (0.23, 0.68), (0.23, 0.63), (0.23, 0.58), (0.23, 0.53), (0.23, 0.48), (0.18, 0.43), (0.18, 0.38), (0.18, 0.33), (0.13, 0.28), (0.08, 0.23), (0.03, 0.18), (-0.02, 0.13), (-0.07, 0.08), (-0.12, 0.03), (-0.17, -0.02), (-0.22, -0.07), (-0.27, -0.12), (-0.32, -0.17), (-0.37, -0.22), (-0.37, -0.27), (-0.37, -0.32), (-0.37, -0.37), (-0.42, -0.42), (-0.42, -0.47), (-0.42, -0.52), (-0.47, -0.57), (-0.52, -0.62), (-0.57, -0.67), (-0.62, -0.72), (-0.67, -0.72), (-0.72, -0.77), (-0.77, -0.77), (-0.82, -0.82), (-0.87, -0.82), (-0.92, -0.87), (-0.97, -0.87)])


    # PATH 4
    start_positions.append((1.1, -0.8))
    goal_positions.append((-1, -0.9))
    reference_paths_map.append([(1.13, -0.77), (1.08, -0.77), (1.03, -0.77), (0.98, -0.77), (0.93, -0.77), (0.88, -0.77), (0.83, -0.77), (0.78, -0.77), (0.73, -0.77), (0.68, -0.77), (0.63, -0.77), (0.58, -0.77), (0.53, -0.77), (0.48, -0.77), (0.43, -0.77), (0.38, -0.77), (0.33, -0.77), (0.28, -0.77), (0.23, -0.77), (0.18, -0.77), (0.13, -0.77), (0.08, -0.77), (0.03, -0.77), (-0.02, -0.77), (-0.07, -0.77), (-0.12, -0.77), (-0.17, -0.77), (-0.22, -0.77), (-0.27, -0.77), (-0.32, -0.77), (-0.37, -0.77), (-0.42, -0.77), (-0.47, -0.77), (-0.52, -0.77), (-0.57, -0.77), (-0.62, -0.77), (-0.67, -0.77), (-0.72, -0.77), (-0.77, -0.82), (-0.82, -0.82), (-0.87, -0.82), (-0.92, -0.87), (-0.97, -0.87)])

    # PATH 5
    start_positions.append((-0.6, 0))
    goal_positions.append((1.1, -0.8))
    reference_paths_map.append([(-0.57, 0.03), (-0.52, 0.03), (-0.47, 0.03), (-0.42, 0.03), (-0.37, 0.03), (-0.32, 0.03), (-0.27, 0.03), (-0.22, -0.02), (-0.17, -0.07), (-0.12, -0.07), (-0.07, -0.07), (-0.02, -0.12), (0.03, -0.17), (0.08, -0.22), (0.13, -0.22), (0.18, -0.27), (0.23, -0.27), (0.28, -0.32), (0.33, -0.32), (0.38, -0.37), (0.43, -0.42), (0.48, -0.42), (0.53, -0.42), (0.58, -0.47), (0.63, -0.52), (0.68, -0.57), (0.73, -0.57), (0.78, -0.57), (0.83, -0.62), (0.88, -0.67), (0.93, -0.67), (0.98, -0.72), (1.03, -0.77), (1.08, -0.77), (1.13, -0.77)])
    serv_start = False
    try:
        rospy.wait_for_service('plan_path_publish', 10)
        plan_path = rospy.ServiceProxy('plan_path_publish', PlanPath)
        serv_start = True
    except Exception as e: 
        planner_proc.terminate()
        rospy.loginfo('Exception caught') 
        return
        
    if serv_start: 

        num_successful_evals = 0
        for test_path_index in range(len(start_positions)):
            rospy.sleep(sleepbetweentrials)
            rospy.loginfo('Test ' + str(test_path_index + 1) + ':')
            try:
                sx = start_positions[test_path_index][0]
                sy = start_positions[test_path_index][1]
                gx = goal_positions[test_path_index][0]
                gy = goal_positions[test_path_index][1]
                ref_path_map = reference_paths_map[test_path_index]

                req = PlanPathRequest(Pose2D(sx,sy,0),Pose2D(gx,gy,0))
                path = plan_path(req).path

                messages, success = check_path(evaluator, gx, gy, "", path, ref_path_map, sx, sy)
                rospy.loginfo(messages)

                if not success:
                    rospy.logwarn('Path is not valid!')
                else:
                    rospy.loginfo('SUCCESS! Path is valid!')
                    num_successful_evals += 1

                path_array = []
                for point in path:
                    path_array.append((np.round(point.x,2), np.round(point.y,2)))
                rospy.loginfo('Planner script output path:')
                rospy.loginfo(path_array)

            except Exception as e:
                rospy.loginfo('[EVALUATION] Exception caught: %s.', e)

        rospy.loginfo("[EVALUATION] Total evaluation success rate: %.2f / %.2f", num_successful_evals, len(start_positions)) 
        # rospy.sleep(60)

    rospy.loginfo("[EVALUATION] Your node's console output has been written to " + planner_script_output_file + " in your current directory.")
    rospy.loginfo("[EVALUATION] Waiting for %.2f seconds and exiting", endsleeptime) 
    rospy.loginfo("[EVALUATION] You can change the sleep time between trials and at the end in this script") 
    rospy.sleep(endsleeptime)

    scan_timer1.cancel()
    scan_timer2.cancel()
    scan_timer3.cancel()
        
    planner_proc.terminate()
    # rospy.sleep(3)
    evaluator.shutdown()
    player_proc.terminate()
    rviz_proc.terminate()
    roscore_proc.terminate()
    

if __name__ == '__main__':
    main()
