#!/usr/bin/env python3

import numpy as np
import rospy
from scipy.ndimage import grey_dilation
from nav_msgs.msg import OccupancyGrid, MapMetaData
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Pose2D
import tf2_ros
from aro_msgs.srv import GenerateFrontier, GenerateFrontierRequest, GenerateFrontierResponse
from aro_msgs.srv import PlanPath, PlanPathRequest, PlanPathResponse
from typing import Optional
from aro_exploration.utils import map_to_grid_coordinates, grid_to_map_coordinates, get_circular_dilation_footprint


invalid_pose = Pose2D(np.nan, np.nan, np.nan)

class Frontier:
    def __init__(self, cell_positions = []):
        self.cell_positions = cell_positions

    def size(self):
        return len(self.cell_positions)

    def get_cell_closest_to_center(self):
        grid_pos = np.array([0,0])

        # TODO: select the frontier cell that is closest to the mean position of this frontier's cells and return its grid coordinates

        return grid_pos 


class FrontierExplorer:
    def __init__(self):
        self.map_frame = rospy.get_param("~map_frame", "icp_map")
        self.robot_frame = rospy.get_param("~robot_frame", "base_footprint")
        self.robot_diameter = float(rospy.get_param("~robot_diameter", 0.6))
        self.min_frontier_size = rospy.get_param("~min_frontier_size", 3)
        self.occupancy_threshold = int(rospy.get_param("~occupancy_threshold", 90))

        self.frontiers = [] # We will store the frontiers to avoid recomputing them if multiple goal requests come before receiving a new grid
        self.is_grid_ready = False # Helper variable to determine if grid was received at least once
        self.are_frontiers_stale = True  # True if the frontiers were not recomputed for the newest grid

        # You may wish to listen to the transformations of the robot
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        # Path planner can be utilized for improved distance heuristics
        #rospy.wait_for_service('plan_path')
        #self.plan_path = rospy.ServiceProxy('plan_path', PlanPath)

        self.vis_pub = rospy.Publisher('frontier_vis', MarkerArray, queue_size=2)
        self.vis_map_pub = rospy.Publisher('frontier_map', OccupancyGrid, queue_size=2)

        # The services and other variables are initialized once an occupancy grid has been received
        self.grf_service: Optional[rospy.ServiceProxy] = None
        self.gcf_service: Optional[rospy.ServiceProxy] = None
        self.gbv_service: Optional[rospy.ServiceProxy] = None
        self.grg_service: Optional[rospy.ServiceProxy] = None


        self.robot_grid_position: Optional[np.ndarray] = None
        self.grid_resolution: Optional[float] = None
        self.occupancy_grid: Optional[np.ndarray] = None
        self.grid_origin_pos: Optional[np.ndarray] = None
        self.grid_info: Optional[MapMetaData] = None

        # TODO: for the semestral work, you can use the visible_occupancy grid to treat walls unseen by the camera as frontiers to force robot too search for the april tag
        self.visible_occupancy_grid: Optional[np.ndarray] = None
        self.vis_occ_grid_origin_pos: Optional[np.ndarray] = None
        self.vis_occ_grid_info: Optional[MapMetaData] = None

        # Subscribe to grids
        self.grid_subscriber = rospy.Subscriber('occupancy', OccupancyGrid, self.grid_cb)
        self.vis_grid_subscriber = rospy.Subscriber('visible_occupancy', OccupancyGrid, self.vis_grid_cb)

    def find_frontiers(self):
        """ Find and store frontiers by running Wavefront Frontier Detection on an inflated grid """
        rospy.loginfo('Finding frontiers')
        self.frontiers = [] # clear any previously computed frontiers

        # update the robot's position on the grid based on current map and transforms
        self.update_robot_grid_coordinates()
        if self.update_robot_grid_coordinates is None:
            rospy.logerr("Could not find frontiers because the robot position could not be retrieved!")
            return
        robot_grid_pos = self.robot_grid_position
        
        # get the kernel - a binary mask, corresponding to the robot shape (or larger for more safety), which will be used for inflating obstacles and unknown space
        dilation_footprint = get_circular_dilation_footprint(self.robot_diameter, self.grid_resolution)

        # TODO: Copy the occupancy grid into some temporary variable(s) and inflate the obstacles and unknown spaces using 'grey_dilation()'

        # TODO: Careful, unknown space might also be an obstacle, and obstacles have priority! Merge inflated grids of occupied and unknown tiles afterwards in right order.

        # TODO: Run the Wavefront Frontier Detection algorithm on the modified occupancy grid - see the presentation slides for details on how to implement it

        # TODO: Also treat the cells in the dilation_footprint centered on the starting position as traversable while doing the BFS to avoid finding 0 frontiers when the robot ends up close to obstacles (see courseware for more info on this)
        
        # TODO: At the end, self.frontiers should contain a list of Frontier objects for all the frontiers you find using the WFD algorithm

        # Extract all the frontiers' cells and their centers to visualize them
        frontier_cells_coords_list = []
        frontier_centers = []
        for frontier in self.frontiers:
            for cell in frontier.cell_positions:
                frontier_cells_coords_list.append(cell)
            frontier_centers.append(frontier.get_cell_closest_to_center())

        # TODO: implement the grid_to_map_coordinates() function to visualize frontiers and anything else you want on the grid (e.g. the inflated grid)
        self.publish_grid_cells_vis(frontier_cells_coords_list, [0, 1, 0, 1], "cells")
        self.publish_grid_cells_vis(frontier_centers, [1, 0, 0, 1], "centers", 0.1)
        if len(self.frontiers) == 0:
            rospy.logwarn("No frontier found.")

    def get_closest_frontier(self, request: GenerateFrontierRequest) -> GenerateFrontierResponse:
        """ Return frontier closest to the robot """
        rospy.loginfo('Finding closest frontier.')
        if self.are_frontiers_stale:
            self.find_frontiers()

        if len(self.frontiers) == 0:
            return GenerateFrontierResponse(invalid_pose)

        closest_frontier_center_grid_pos = np.array([0,0])

        # TODO: find the frontier center that is closest to the robot

        # convert from grid coordinates to map coordinates and return service response
        map_x, map_y = grid_to_map_coordinates(closest_frontier_center_grid_pos, self.grid_info)
        response = GenerateFrontierResponse(Pose2D(map_x, map_y, 0.0))
        rospy.loginfo('Returning closest frontier center at ' + str(map_x) + ' ' + str(map_y))
        return response

    def get_best_value_frontier(self, request: GenerateFrontierRequest) -> GenerateFrontierResponse:
        """ Return best frontier """
        rospy.loginfo('Finding best value frontier.')
        if self.are_frontiers_stale:
            self.find_frontiers()

        if len(self.frontiers) == 0:
            return GenerateFrontierResponse(invalid_pose)

        best_frontier_center_grid_pos = np.array([0,0])

        # TODO: for semestral work - find the best frontier to explore, based on your designed value function (largest, closest, something in between?)

        # convert from grid coordinates to map coordinates and return response
        map_x, map_y = grid_to_map_coordinates(best_frontier_center_grid_pos, self.grid_info)
        response = GenerateFrontierResponse(Pose2D(map_x, map_y, 0.0))
        rospy.loginfo('Returning best frontier at ' + str(map_x) + ' ' + str(map_y))
        return response

    def update_robot_grid_coordinates(self):
        """ Get the current robot position in the grid """
        try:
            trans_msg = self.tf_buffer.lookup_transform(self.map_frame, self.robot_frame, rospy.Time(), rospy.Duration(0.5))
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
            rospy.logwarn("Failed to get the transform from "+self.map_frame+' to '+self.robot_frame + ' to update robot position.')
            self.robot_grid_position = None
        else:
            map_pos = np.array([trans_msg.transform.translation.x, trans_msg.transform.translation.y])
            self.robot_grid_position = map_to_grid_coordinates(map_pos, self.grid_info).astype(int)

        # visualize which cell the robot's center currently lies in. Will be at (0,0) until you implement map_to_grid_coordinates
        self.publish_grid_cells_vis([self.robot_grid_position], [0, 0, 1, 1], "robot_pos", 0.2)

    def extract_grid(self, msg):
        width, height, self.grid_resolution = msg.info.width, msg.info.height, msg.info.resolution
        self.grid_origin_pos = np.array([msg.info.origin.position.x, msg.info.origin.position.y])
        self.grid_info = msg.info
        self.occupancy_grid = np.reshape(msg.data, (height, width))

    def grid_cb(self, msg):
        self.extract_grid(msg)
        self.are_frontiers_stale = True
        if not self.is_grid_ready:


            # Create services
            self.gcf_service = rospy.Service('get_closest_frontier', GenerateFrontier, self.get_closest_frontier)
            self.gbv_service = rospy.Service('get_best_value_frontier', GenerateFrontier, self.get_best_value_frontier)
            self.is_grid_ready = True

    def extract_vis_grid(self, msg):
        width, height = msg.info.width, msg.info.height
        self.vis_occ_grid_origin_pos = np.array([msg.info.origin.position.x, msg.info.origin.position.y])        
        self.vis_occ_grid_info: msg.info
        self.visible_occupancy_grid = np.reshape(msg.data, (height, width)) 

    def vis_grid_cb(self, msg):
        self.extract_vis_grid(msg)

    def publish_grid_cells_vis(self, grid_positions, clr, rviz_namespace = "default", z_pos = 0 ):
        """ Visualize given cells of the grid """
        # grid_positions: a list of [cell_x, cell_y] coordinates
        # clr: marker color as [r,g,b,a] with values from 0 to 1
        # rviz_namespace: use to visualize multiple things under one topic, can toggle them in rviz
        # z_pos: the camera looks top_down, so higher z_pos puts these markers on top

        marker_array = MarkerArray()
        marker_id = 0
        for i in range(len(grid_positions)):
            marker = Marker()
            marker.ns = rviz_namespace
            marker.id = marker_id
            marker_id += 1
            marker.header.frame_id = self.map_frame
            marker.type = Marker.CUBE
            marker.action = 0
            marker.scale.x = self.grid_resolution
            marker.scale.y = self.grid_resolution
            marker.scale.z = self.grid_resolution
            # x, y = grid_to_map_coordinates(np.array([f.points[i][1], f.points[i][0]]), self.grid_info)
            map_x, map_y = grid_to_map_coordinates(grid_positions[i], self.grid_info)
            marker.pose.position.x = map_x
            marker.pose.position.y = map_y
            marker.pose.position.z = z_pos
            marker.pose.orientation.x = 0
            marker.pose.orientation.y = 0
            marker.pose.orientation.z = 0
            marker.pose.orientation.w = 1
            marker.color.r = clr[0]
            marker.color.g = clr[1]
            marker.color.b = clr[2]
            marker.color.a = clr[3]
            marker_array.markers.append(marker)
        self.vis_pub.publish(marker_array)



if __name__ == "__main__":
    rospy.init_node("frontier_explorer")
    fe = FrontierExplorer()
    rospy.spin()
