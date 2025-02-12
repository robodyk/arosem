cmake_minimum_required(VERSION 3.16.3)
project(aro_msgs)

find_package(catkin REQUIRED COMPONENTS
  actionlib_msgs
  geometry_msgs
  message_generation
  std_msgs
)

add_message_files(
  FILES
  Path.msg
  SectorDistances.msg
)

add_service_files(
  FILES
  GenerateFrontier.srv
  PlanPath.srv
)

add_action_files(
  FILES
  FollowPath.action
)

generate_messages(
  DEPENDENCIES
  actionlib_msgs
  geometry_msgs
  std_msgs
)

catkin_package(
  CATKIN_DEPENDS
    actionlib_msgs geometry_msgs message_runtime std_msgs
)
