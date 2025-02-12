#!/usr/bin/env python3
"""
Simple reactive controller for turtlebot robot.
"""

import rospy
import numpy as np  # you probably gonna need this
from geometry_msgs.msg import Twist, Vector3
from aro_msgs.msg import SectorDistances
from std_srvs.srv import SetBool, SetBoolRequest, SetBoolResponse

# TODO HW 01 and HW 02: add necessary imports

class ReactiveController():

    def __init__(self):
        rospy.loginfo('Initializing node')
        rospy.init_node('reactive_controller')
        self.initialized = False

        # TODO HW 01: register listener for laser scan message

        # TODO HW 02:publisher for "/cmd_vel" topic (use arg "latch=True")

        # TODO HW 01: register publisher for "/reactive_control/sector_dists" topic
        # publishing minimum distances as message of type aro_msgs/SectorDistances

        # TODO HW 02: create proxy for the mission evaluation service and wait till the start service is up

        # TODO HW 02: create service server for mission start

        # TODO: you are probably going to need to add some variables

        # TODO HW 02: add timer for mission end checking

        self.initialized = True

        rospy.loginfo('Reactive controller initialized. Waiting for data.')

    def timer_cb(self, event): 
        """
        Callback function for timer.

        :param event (rospy TimerEvent): event handled by the callback 
        """
       
        if not self.initialized:
            return

        # TODO HW 02: Check that the active time had elapsed and send the mission evaluation request


    def activate_cb(self, req: SetBoolRequest) -> SetBoolResponse: 
        """
        Activation service callback.

        :param req: obtained ROS service request 

        :return: ROS service response 
        """

        rospy.loginfo_once('Activation callback entered')

        # TODO HW 02: Implement callback for activation service
        # Do not forget that this is a service callback, so you have to return a SetBoolResponse object with .success
        # set to True.

        pass

    def scan_cb(self, msg):
        """
        Scan callback.

        :param msg: obtained message with data from 2D scanner of type ??? 
        """
        rospy.loginfo_once('Scan callback entered')

        if not self.initialized: 
            return

        # TODO HW 01: Implement callback for 2D scan, process and publish required data

        # TODO HW 02: Add robot control based on received scan


    def apply_control_inputs(self, velocity: float, angular_rate: float):
        """
        Applies control inputs.

        :param velocity: required forward velocity of the robot 
        :param angular_rate: required angular rate of the robot 
        """
        # TODO HW 02: publish required control inputs 

        pass

if __name__ == '__main__':
    rc = ReactiveController()
    rospy.spin()
