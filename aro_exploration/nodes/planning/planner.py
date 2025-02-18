#!/usr/bin/env python3

from __future__ import absolute_import, division, print_function
import rospy
import numpy as np
from queue import PriorityQueue
from typing import List
from scipy.ndimage import grey_dilation
from nav_msgs.msg import OccupancyGrid, MapMetaData, Path
from geometry_msgs.msg import Pose2D, Pose, PoseStamped, Point, Quaternion
from visualization_msgs.msg import MarkerArray, Marker
from aro_msgs.srv import PlanPath, PlanPathRequest, PlanPathResponse
import tf2_ros
from aro_exploration.utils import map_to_grid_coordinates, grid_to_map_coordinates, get_circular_dilation_footprint


"""
Here are imports that you will most likely need. However, you may wish to remove or add your own import.
"""

class PathPlanner:
    def __init__(self):
        # Initialize the node
        rospy.init_node("path_planner")

        self.robot_grid_position = None
        self.grid = None
        self.grid_info = None
        self.origin_pos = None
        self.grid_resolution = None
        # Helper variable to determine if grid was received at least once
        self.grid_ready = False

        self.map_frame = rospy.get_param("~map_frame", "icp_map")
        self.robot_frame = rospy.get_param("~robot_frame", "base_footprint")
        self.robot_diameter = float(rospy.get_param("~robot_diameter", 0.6))
        self.occupancy_threshold = int(rospy.get_param("~occupancy_threshold", 25))

        # You may wish to listen to the transformations of the robot
        self.tf_buffer = tf2_ros.Buffer()
        # Use the tfBuffer to obtain transformation as needed
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        # Publishers for visualization
        self.path_vis_pub = rospy.Publisher('path', Path, queue_size=1)
        self.start_and_goal_vis_pub = rospy.Publisher('start_and_goal', MarkerArray, queue_size=1)

        # The services will be set up when the first occupancy grid message comes
        self.plan_publish_service = None
        self.plan_service = None

        # Subscribe to grid
        self.grid_subscriber = rospy.Subscriber('occupancy', OccupancyGrid, self.grid_cb)

        rospy.loginfo('Path planner initialized.')

    def plan_path_and_publish(self, request: PlanPathRequest):
        response = self.plan_path(request)
        self.publish_path(response.path)
        return response

    def plan_path(self, request: PlanPathRequest) -> PlanPathResponse:
        """ Plan and return path from the requrested start position to the requested goal """

        # Get the position of the goal (real-world)
        start_position = map_to_grid_coordinates(np.array([request.start.x, request.start.y]), self.grid_info)
        goal_position = map_to_grid_coordinates(np.array([request.goal.x, request.goal.y]), self.grid_info)

        # Visualize start and goal
        self.publish_start_and_goal_vis(request.start, request.goal)

        # check that start and goal positions are inside the grid. 
        if start_position[0] < 0 or start_position[1] < 0 or start_position[0] >= self.grid.shape[1] or start_position[1] >= self.grid.shape[0]:
            rospy.logwarn(
                "WARNING: start grid position is outside the grid. [cell_x,cell_y]=[{:f},{:f}], grid shape [w={:f},h={:f}]. "
                "Returning an empty trajectory.".format(
                    start_position[0], start_position[1], self.grid.shape[1], self.grid.shape[0]))
            response = PlanPathResponse([])
            return response

        if goal_position[0] < 0 or goal_position[1] < 0 or goal_position[0] >= self.grid.shape[1] or goal_position[1] >= self.grid.shape[0]:
            rospy.logwarn(
                "WARNING: goal grid position is outside the grid. [cell_x,cell_y]=[{:f},{:f}], grid shape [w={:f},h={:f}]. "
                "Returning an empty trajectory.".format(
                    goal_position[0], goal_position[1], self.grid.shape[1], self.grid.shape[0]))
            response = PlanPathResponse([])
            return response



        path = []
        dilation_footprint = get_circular_dilation_footprint(self.robot_diameter, self.grid_resolution)

        # TODO: Copy the occupancy grid into some temporary variable and inflate the obstacles. Use the kernel from get_circular_dilation_footprint()

        # TODO: Make sure you take into account unknown grid tiles as non-traversable and also inflate those.

        # TODO: Compute the path using A* and a euclidean distance heuristic. You can move diagonally, but check the safety of the 4 cells 

        # TODO: Grid tiles could and should be explored repeatedly with updated cost.

        # TODO: Hint: utilize look-up grids for bools and cost values, storing visited points in list is SLOW. Also consider using an ordered queue (e.g. PriorityQueue)

        # TODO: Submitting badly optimized code can lead to failed evaluations during semestral work simulation.

        # Convert the path (list of grid cell coordinates) into a service response with a list of /icp_map frame poses 
        real_path = [Pose2D(pos[0], pos[1], 0) for pos in
                     [grid_to_map_coordinates(waypoint, self.grid_info) for waypoint in path]]
        response = PlanPathResponse(real_path)



        return response

    def extract_grid(self, msg):
        width, height, self.grid_resolution = msg.info.width, msg.info.height, msg.info.resolution
        self.origin_pos = np.array([msg.info.origin.position.x, msg.info.origin.position.y])
        self.grid_info = msg.info
        self.grid = np.reshape(msg.data, (height, width))

    def grid_cb(self, msg):
        self.extract_grid(msg)
        if not self.grid_ready:
            # Create services
            self.plan_publish_service = rospy.Service('plan_path_publish', PlanPath, self.plan_path_and_publish)
            self.plan_service = rospy.Service('plan_path', PlanPath, self.plan_path)
            self.grid_ready = True

    def publish_path(self, path_2d: List[Pose2D]):
        msg = Path()
        msg.header.frame_id = self.map_frame
        msg.header.stamp = rospy.get_rostime()
        for waypoint in path_2d:
            pose = PoseStamped()
            pose.header.frame_id = self.map_frame
            pose.pose.position.x = waypoint.x
            pose.pose.position.y = waypoint.y
            pose.pose.position.z = 0
            msg.poses.append(pose)

        rospy.loginfo("Publishing plan.")
        self.path_vis_pub.publish(msg)

    def publish_start_and_goal_vis(self, start: Pose2D, goal: Pose2D):
        msg = MarkerArray()
        m_start = Marker()
        m_start.header.frame_id = self.map_frame
        m_start.id = 1
        m_start.type = 2
        m_start.action = 0
        m_start.pose = Pose()
        m_start.pose.orientation = Quaternion(0.0, 0.0, 0.0, 1.0)
        m_start.pose.position = Point(start.x, start.y, 0.0)
        # m_start.points.append(Point(start.x, start.y, 0.0))
        m_start.color.r = 1.0
        m_start.color.g = 0.0
        m_start.color.b = 0.0
        m_start.color.a = 0.8
        m_start.scale.x = 0.1
        m_start.scale.y = 0.1
        m_start.scale.z = 0.001
        msg.markers.append(m_start)

        # goal marker
        m_goal = Marker()
        m_goal.header.frame_id = self.map_frame
        m_goal.id = 2
        m_goal.type = 2
        m_goal.action = 0
        m_goal.pose = Pose()
        m_goal.pose.orientation = Quaternion(0.0, 0.0, 0.0, 1.0)
        m_goal.pose.position = Point(goal.x, goal.y, 0.0) 
        # m_start.points.append(Point(start.x, start.y, 0.0))
        m_goal.color.r = 0.0
        m_goal.color.g = 1.0
        m_goal.color.b = 0.0
        m_goal.color.a = 0.8
        m_goal.scale.x = 0.1
        m_goal.scale.y = 0.1
        m_goal.scale.z = 0.001
        msg.markers.append(m_goal)
        rospy.loginfo("Publishing start and goal markers.")
        self.start_and_goal_vis_pub.publish(msg)


if __name__ == "__main__":
    pp = PathPlanner()

    rospy.spin()
