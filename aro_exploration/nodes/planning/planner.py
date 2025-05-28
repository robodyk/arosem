#!/usr/bin/env python3

from queue import PriorityQueue
from typing import List

import numpy as np
import rospy
import tf2_ros
from geometry_msgs.msg import Point, Pose, Pose2D, PoseStamped, Quaternion
from nav_msgs.msg import MapMetaData, OccupancyGrid, Path
from scipy.ndimage import grey_dilation
from visualization_msgs.msg import Marker, MarkerArray

from aro_exploration.utils import (
    get_circular_dilation_footprint,
    grid_to_map_coordinates,
    map_to_grid_coordinates,
)
from aro_msgs.srv import PlanPath, PlanPathRequest, PlanPathResponse

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
        self.path_vis_pub = rospy.Publisher("path", Path, queue_size=1)
        self.start_and_goal_vis_pub = rospy.Publisher(
            "start_and_goal",
            MarkerArray,
            queue_size=1,
        )

        # The services will be set up when the first occupancy grid message comes
        self.plan_publish_service = None
        self.plan_service = None

        # Subscribe to grid
        self.grid_subscriber = rospy.Subscriber(
            "occupancy",
            OccupancyGrid,
            self.grid_cb,
        )

        rospy.loginfo("Path planner initialized.")

    def plan_path_and_publish(self, request: PlanPathRequest):
        response = self.plan_path(request)
        self.publish_path(response.path)
        return response

    def plan_path(self, request: PlanPathRequest) -> PlanPathResponse:
        """Plan and return path from the requrested start position to the requested goal"""
        # Get the position of the goal (real-world)
        start_position = map_to_grid_coordinates(
            np.array([request.start.x, request.start.y]),
            self.grid_info,
        )
        goal_position = map_to_grid_coordinates(
            np.array([request.goal.x, request.goal.y]),
            self.grid_info,
        )

        # Visualize start and goal
        self.publish_start_and_goal_vis(request.start, request.goal)

        # check that start and goal positions are inside the grid.
        if (
            start_position[0] < 0
            or start_position[1] < 0
            or start_position[0] >= self.grid.shape[1]
            or start_position[1] >= self.grid.shape[0]
        ):
            rospy.logwarn(
                f"WARNING: start grid position is outside the grid. [cell_x,cell_y]=[{start_position[0]:f},{start_position[1]:f}], grid shape [w={self.grid.shape[1]:f},h={self.grid.shape[0]:f}]. "
                "Returning an empty trajectory.",
            )
            response = PlanPathResponse([])
            return response

        if (
            goal_position[0] < 0
            or goal_position[1] < 0
            or goal_position[0] >= self.grid.shape[1]
            or goal_position[1] >= self.grid.shape[0]
        ):
            rospy.logwarn(
                f"WARNING: goal grid position is outside the grid. [cell_x,cell_y]=[{goal_position[0]:f},{goal_position[1]:f}], grid shape [w={self.grid.shape[1]:f},h={self.grid.shape[0]:f}]. "
                "Returning an empty trajectory.",
            )
            response = PlanPathResponse([])
            return response

        # Get circular dilation footprint for obstacle inflation
        dilation_footprint = get_circular_dilation_footprint(
            self.robot_diameter,
            self.grid_resolution,
        )

        # Copy the occupancy grid to a temporary variable
        # Convert to binary map where:
        # 0 = free space (< occupancy_threshold)
        # 1 = occupied or unknown space (>= occupancy_threshold or == -1)
        temp_grid = np.zeros_like(self.grid, dtype=np.uint8)
        temp_grid[self.grid >= self.occupancy_threshold] = 1  # Occupied space
        temp_grid[self.grid == -1] = 1  # Unknown space

        # Inflate obstacles using the dilation footprint
        inflated_grid = grey_dilation(temp_grid, footprint=dilation_footprint)

        # Clear the space around the start position to avoid getting stuck
        start_x, start_y = int(start_position[0]), int(start_position[1])

        # Create a temporary grid for the start position dilation
        start_pos_grid = np.zeros_like(self.grid, dtype=np.uint8)
        start_pos_grid[start_y, start_x] = 1

        # Dilate the start position
        dilated_start = grey_dilation(start_pos_grid, footprint=dilation_footprint)

        # Clear the space (set to 0) for cells that are in the dilated start region
        inflated_grid[dilated_start > 0] = 0

        # Check if goal is in obstacle space
        if inflated_grid[int(goal_position[1]), int(goal_position[0])] == 1:
            rospy.logwarn(
                "WARNING: goal position is in obstacle space or too close to obstacles. "
                "Returning an empty trajectory.",
            )
            response = PlanPathResponse([])
            return response

        # A* algorithm
        # Initialize data structures
        open_set = PriorityQueue()
        came_from = {}

        # Cost from start to current node
        g_score = {}
        # For the grid, we'll use tuples of (x, y) as keys
        start = (int(start_position[0]), int(start_position[1]))
        goal = (int(goal_position[0]), int(goal_position[1]))

        # Initialize g_score with infinity for all cells
        for y in range(inflated_grid.shape[0]):
            for x in range(inflated_grid.shape[1]):
                g_score[(x, y)] = float("inf")

        g_score[start] = 0

        # Estimated total cost from start to goal through current node
        f_score = g_score.copy()

        # Heuristic function (Euclidean distance)
        def heuristic(pos, goal):
            return np.sqrt((pos[0] - goal[0]) ** 2 + (pos[1] - goal[1]) ** 2)

        f_score[start] = heuristic(start, goal)

        # Add start node to open set with its f_score as priority
        open_set.put((f_score[start], start))

        # To keep track of nodes in the open set
        in_open_set = {start}

        # Possible movement directions (8-connected grid: horizontal, vertical, and diagonal)
        directions = [
            (0, 1),
            (1, 0),
            (0, -1),
            (-1, 0),  # Horizontal and vertical
            (1, 1),
            (1, -1),
            (-1, -1),
            (-1, 1),  # Diagonal
        ]

        # Cost of movement
        cost = {
            (0, 1): 1,
            (1, 0): 1,
            (0, -1): 1,
            (-1, 0): 1,  # Horizontal and vertical cost = 1
            (1, 1): 1.414,
            (1, -1): 1.414,
            (-1, -1): 1.414,
            (-1, 1): 1.414,  # Diagonal cost = sqrt(2)
        }

        while not open_set.empty():
            # Get the node with the lowest f_score
            _, current = open_set.get()
            in_open_set.discard(current)

            # If we reached the goal, reconstruct and return the path
            if current == goal:
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                path.reverse()

                # Convert the path to the map coordinates
                real_path = [
                    Pose2D(pos[0], pos[1], 0)
                    for pos in [
                        grid_to_map_coordinates(np.array(waypoint), self.grid_info)
                        for waypoint in path
                    ]
                ]
                real_path[-1].theta = request.goal.theta
                return PlanPathResponse(real_path)

            # Check all neighbors
            for dx, dy in directions:
                neighbor = (current[0] + dx, current[1] + dy)

                # Check if neighbor is within grid boundaries
                if (
                    0 <= neighbor[0] < inflated_grid.shape[1]
                    and 0 <= neighbor[1] < inflated_grid.shape[0]
                ):
                    # For diagonal movement, check if the four adjacent cells are free
                    if abs(dx) == 1 and abs(dy) == 1:
                        # Check the two adjacent cells that would be cut through in diagonal movement
                        if (
                            inflated_grid[current[1], current[0] + dx] == 1
                            or inflated_grid[current[1] + dy, current[0]] == 1
                        ):
                            continue

                    # Check if neighbor is free space
                    if inflated_grid[neighbor[1], neighbor[0]] == 0:
                        # Calculate tentative g_score
                        tentative_g_score = g_score[current] + cost[(dx, dy)]

                        if tentative_g_score < g_score[neighbor]:
                            # This path to neighbor is better than any previous one
                            came_from[neighbor] = current
                            g_score[neighbor] = tentative_g_score
                            f_score[neighbor] = tentative_g_score + heuristic(
                                neighbor,
                                goal,
                            )

                            if neighbor not in in_open_set:
                                open_set.put((f_score[neighbor], neighbor))
                                in_open_set.add(neighbor)

        # If we get here, there's no path to the goal
        rospy.logwarn("WARNING: No path found from start to goal.")
        return PlanPathResponse([])

    def extract_grid(self, msg):
        width, height, self.grid_resolution = (
            msg.info.width,
            msg.info.height,
            msg.info.resolution,
        )
        self.origin_pos = np.array(
            [msg.info.origin.position.x, msg.info.origin.position.y],
        )
        self.grid_info = msg.info
        self.grid = np.reshape(msg.data, (height, width))

    def grid_cb(self, msg):
        self.extract_grid(msg)
        if not self.grid_ready:
            # Create services
            self.plan_publish_service = rospy.Service(
                "plan_path_publish",
                PlanPath,
                self.plan_path_and_publish,
            )
            self.plan_service = rospy.Service("plan_path", PlanPath, self.plan_path)
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
