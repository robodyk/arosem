#!/usr/bin/env python3
"""
Simple reactive controller for turtlebot robot.
"""

import numpy as np  # you probably gonna need this
import rospy
from aro_msgs.msg import SectorDistances
from geometry_msgs.msg import Twist

# TODO HW 01 and HW 02: add necessary imports
from sensor_msgs.msg import LaserScan
from std_srvs.srv import SetBool, SetBoolRequest, SetBoolResponse, Trigger


class ReactiveController:
    def __init__(self):
        rospy.loginfo("Initializing node")
        rospy.init_node("reactive_controller")
        self.initialized = False

        # TODO HW 01: register listener for laser scan message
        self.scan_subscriber = rospy.Subscriber("/scan", LaserScan, self.scan_cb)

        # TODO HW 02:publisher for "/cmd_vel" topic (use arg "latch=True")
        self.vel_publisher = rospy.Publisher("/cmd_vel", Twist, latch=True)

        # TODO HW 01: register publisher for "/reactive_control/sector_dists" topic
        # publishing minimum distances as message of type aro_msgs/SectorDistances
        self.dists_publisher = rospy.Publisher(
            "/reactive_control/sector_dists",
            SectorDistances,
            queue_size=10,
        )

        # TODO HW 02: create proxy for the mission evaluation service and wait till the start service is up
        self.eval_service_proxy = rospy.ServiceProxy(
            "/reactive_control/evaluate_mission",
            Trigger,
        )

        # TODO HW 02: create service server for mission start
        self.reactive_control_server = rospy.Service(
            "/reactive_control/activate",
            SetBool,
            self.activate_cb,
        )

        # TODO: you are probably going to need to add some variables

        # TODO HW 02: add timer for mission end checking
        self.reactive_control_timer = rospy.Timer(rospy.Duration(1), self.timer_cb)

        self.reactive_control_activated = False
        self.movement_activated = False

        self.initialized = True

        self.time_activated = rospy.Time.now()

        rospy.loginfo("Reactive controller initialized. Waiting for data.")

    def timer_cb(self, event):
        """
        Callback function for timer.

        :param event (rospy TimerEvent): event handled by the callback
        """
        if not (self.initialized and self.reactive_control_activated):
            return

        time = (rospy.Time.now() - self.time_activated).secs
        if time >= 49:
            self.movement_activated = False
            self.reactive_control_activated = False
            self.eval_service_proxy()

        # TODO HW 02: Check that the active time had elapsed and send the mission evaluation request

    def activate_cb(self, req: SetBoolRequest) -> SetBoolResponse:
        """
        Activation service callback.

        :param req: obtained ROS service request

        :return: ROS service response
        """
        rospy.loginfo_once("Activation callback entered")

        if req.data:
            if not self.reactive_control_activated:
                self.time_activated = rospy.Time.now()
            self.reactive_control_activated = True
        self.movement_activated = req.data

        ret = SetBoolResponse()
        ret.success = True

        return ret

        # TODO HW 02: Implement callback for activation service
        # Do not forget that this is a service callback, so you have to return a SetBoolResponse object with .success
        # set to True.

    def scan_cb(self, msg: LaserScan):
        """
        Scan callback.

        :param msg: obtained message with data from 2D scanner of type ???
        """
        rospy.loginfo_once("Scan callback entered")

        if not self.initialized:
            return

        r = np.array(msg.ranges)
        r[np.logical_or(r < msg.range_min, r > msg.range_max)] = np.nan

        ret = SectorDistances()
        ret.distance_right = np.nanmin(r[270:330])
        ret.distance_front = min(np.nanmin(r[330:]), np.nanmin(r[:30]))
        ret.distance_left = np.nanmin(r[30:90])

        self.dists_publisher.publish(ret)

        move_msg = Twist()
        if self.movement_activated:
            if ret.distance_front > 0.2:
                move_msg.linear.x = 0.22

            if ret.distance_front < 0.4:
                s = 1 if ret.distance_right < ret.distance_left else -1
                move_msg.angular.z = s * 0.9 * np.pi

        self.vel_publisher.publish(move_msg)

        # TODO HW 02: Add robot control based on received scan

    def apply_control_inputs(self, velocity: float, angular_rate: float):
        """
        Applies control inputs.

        :param velocity: required forward velocity of the robot
        :param angular_rate: required angular rate of the robot
        """
        # TODO HW 02: publish required control inputs


if __name__ == "__main__":
    rc = ReactiveController()
    rospy.spin()
