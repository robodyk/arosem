#!/usr/bin/env python3
"""
Script for local evaluation of ARO HW 2 - reactive controller.
"""
import rospy
import numpy as np
from geometry_msgs.msg import Twist, Vector3
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float32
from std_srvs.srv import Trigger, SetBool, SetBoolRequest, TriggerResponse, TriggerRequest
from nav_msgs.msg import Odometry
from aro_msgs.msg import SectorDistances

class ReactiveControllerEvaluator():

    def __init__(self):

        rospy.init_node('reactive_controller_evaluator')
        rospy.loginfo('Initializing reactive controller evaluator node.')

        self.initialized = False
        self.activated = False
        self.mission_evaluation_required = False
        self.is_robot_static = True
        self.failure = False

        self.moved_while_deactivated = False
        self.moved_before_activation = False

        self.prev_odom = None
        self.init_pose = None

        self.max_dist_from_start = 0.0
        self.traveled_distance = 0.0

        self.use_manual_control = rospy.get_param('~use_manual_control', False)
        self.save_results = rospy.get_param('~save_results', False)
        self.results_file = rospy.get_param('~results_file', 'data/results.tsxt')
        timer_freq = rospy.get_param('~timer_freq', 2)
        self.time_start = rospy.get_param('~time_start', 5)
        self.time_stop = rospy.get_param('~time_stop', 20)
        self.time_start_2 = rospy.get_param('~time_start_2', 30)
        self.timeout_stop = rospy.get_param('~timeout_stop', 3)
        self.timeout_end = rospy.get_param('~timeout_end', 51)
        self.min_required_dist_from_start = rospy.get_param('~min_required_dist_from_start', 1.5)
        self.min_required_traveled_dist = rospy.get_param('~min_required_traveled_dist', 8.0)
        self.static_dist_threshold = rospy.get_param('~static_dist_threshold', 0.01)
        self.angular_rate_static_threshold = rospy.get_param('~angular_rate_static_threshold', 0.01)

        self.evaluate_mission = rospy.get_param('~evaluate_mission', True)

        self.time_of_evaluation_request = None

        self.srv_start = rospy.Service('/reactive_control/evaluate_mission', Trigger, self.cb_evaluate_mission)

        if not self.use_manual_control:
            self.srv_cl_activate_control = rospy.ServiceProxy('/reactive_control/activate', SetBool)
            while not rospy.is_shutdown():
                try:
                    self.srv_cl_activate_control.wait_for_service(timeout=1.0)
                    rospy.loginfo('Connected to mission activation service.')
                    break
                except rospy.ROSException:
                    rospy.logwarn('Waiting for mission activation service.')

        self.sub_odom = rospy.Subscriber('/ground_truth_odom', Odometry, self.cb_odom)
        self.sub_sector_dists = rospy.Subscriber('/reactive_control/sector_dists', SectorDistances, self.cb_sectors)

        # Initialize publishers for visualizations
        self.pub_traveled_distance = rospy.Publisher('/reactive_control/evaluation/traveled_distance', Float32, queue_size=1)
        self.pub_max_distance_from_start = rospy.Publisher('/reactive_control/evaluation/max_distance_from_start', Float32, queue_size=1)
        self.pub_dist_left = rospy.Publisher('/reactive_control/min_dist_left', Float32, queue_size=1)
        self.pub_dist_front = rospy.Publisher('/reactive_control/min_dist_front', Float32, queue_size=1)
        self.pub_dist_right = rospy.Publisher('/reactive_control/min_dist_right', Float32, queue_size=1)

        self.initialization_time = rospy.get_time()
        
        if not self.use_manual_control:
            self.timer = rospy.Timer(rospy.Duration.from_sec(1.0/timer_freq), self.timer_cb)

        self.initialized = True
        rospy.loginfo('Node initialized.')

    def print_and_save_results(self, service_called):

        if self.save_results:
            rospy.loginfo('Saving evaluation data to %s.', self.results_file)
            f = open(self.results_file, "w")
        
        rospy.loginfo('RESULTS:')
        lines = []

        # check if robot is static at the end
        if not self.is_robot_static:
            lines.append('Robot is static at the end: FAILED')
            rospy.logwarn(lines[-1])
            self.failure = True
        else:
            lines.append('Robot is static at the end: PASSED')
            rospy.loginfo(lines[-1])

        # check if robot is static after call to deactivation service
        if self.moved_before_activation:
            lines.append('Robot is static before activation: FAILED')
            rospy.logwarn(lines[-1])
            self.failure = True
        else:
            lines.append('Robot is static before activation: PASSED')
            rospy.loginfo(lines[-1])

        # check if robot is static while deactivated
        if self.moved_while_deactivated: 
            lines.append('Robot is static after deactivation: FAILED')
            rospy.logwarn(lines[-1])
            self.failure = True
        else:
            lines.append('Robot is static after deactivation: PASSED')
            rospy.loginfo(lines[-1])

        # check maximum distance from start
        if self.max_dist_from_start < self.min_required_dist_from_start:
            lines.append('Minimum distance from start: FAILED (dist. reached: {:.2f} m, dist. required: {:.2f} m)'.format(self.max_dist_from_start, self.min_required_dist_from_start))
            rospy.logwarn(lines[-1])
            self.failure = True
        else:
            lines.append('Minimum distance from start: PASSED (dist. reached: {:.2f} m, dist. required: {:.2f} m)'.format(self.max_dist_from_start, self.min_required_dist_from_start))
            rospy.loginfo(lines[-1])

        # check traveled distance
        if self.traveled_distance < self.min_required_traveled_dist:
            lines.append('Minimum traveled distance: FAILED (dist. traveled: {:.2f} m, dist. required: {:.2f} m)'.format(self.traveled_distance, self.min_required_traveled_dist))
            rospy.logwarn(lines[-1])
            self.failure = True
        else:
            lines.append('Minimum traveled distance: PASSED (dist. traveled: {:.2f} m, dist. required: {:.2f} m)'.format(self.traveled_distance, self.min_required_traveled_dist))
            rospy.loginfo(lines[-1])

        # check if robot is static at the end
        if not service_called:
            lines.append('Evaluation service called: FAILED')
            rospy.logwarn(lines[-1])
            self.failure = True
        else:
            lines.append('Evaluation service called: PASSED')
            rospy.loginfo(lines[-1])

        if self.failure:
            lines.append('Overall status: FAILED')
            rospy.logerr(lines[-1])
        else:
            lines.append('Overall status: PASSED')
            rospy.loginfo(lines[-1])

        if self.save_results: 
            for line in lines: 
                f.write(line)
                f.write('\n')

            f.close()
            rospy.loginfo('Results saved to %s.', self.results_file)
            

    def timer_cb(self, event):
        """
        Callback function for main timer handling the evaluation.

        :param event (rospy TimerEvent): event handled by the callback 
        """
        rospy.loginfo_once('Main timer spinning.')

        time_from_init = rospy.get_time() - self.initialization_time

        # enable continuous motion without deactivation and evaluation
        if not self.evaluate_mission:

            if not self.activated:
                if time_from_init > self.time_start:
                    self.call_control_activation_service(True)

                else:
                    rospy.loginfo('Robot motion will be activated in %.2f s.', self.time_start - time_from_init)

            return

        if self.mission_evaluation_required: 

            if (rospy.get_time() - self.time_of_evaluation_request) < self.timeout_stop:
                rospy.loginfo_once('Waiting for robot to stop the motion.')
            
            else: 

                max_dist_from_start_ok = True
                traveled_distance_ok = True
                static_at_the_end_ok = True

                self.print_and_save_results(True)
                
                self.call_control_activation_service(False)
                self.timer.shutdown() 

        elif time_from_init - self.time_start > self.timeout_end:
            self.failure = True 

            self.print_and_save_results(False)

            self.call_control_activation_service(False)
            self.timer.shutdown()

        else: 

            if not self.activated: 
                if time_from_init < self.time_start:

                    if not self.is_robot_static:
                        self.moved_before_activation = True
                        self.failure = True

                    rospy.loginfo('Robot motion will be activated in %.2f s.', self.time_start - time_from_init)

                elif time_from_init < self.time_stop:

                    if not self.is_robot_static:
                        self.failure = True
                    
                    rospy.loginfo('Activating robot motion.')
                    self.call_control_activation_service(True)

                elif time_from_init < self.time_stop + self.timeout_stop:

                    rospy.loginfo_once('Waiting for robot motion to be stopped.')

                elif time_from_init < self.time_start_2:

                    if not self.is_robot_static:
                        self.moved_while_deactivated = True
                        rospy.logwarn('Robot is not static after deactivation of motion!')
                    rospy.loginfo_throttle(2.5, 'Remaining mission time: %.2f s.', self.timeout_end - time_from_init + self.time_start)

                else:
                    rospy.loginfo('Activating robot motion for the second time.')
                    self.call_control_activation_service(True)

            else:
                remaining_mission_time = self.timeout_end - time_from_init + self.time_start
                if remaining_mission_time > 10.0: 
                    rospy.loginfo_throttle(2.5, 'Remaining mission time: %.2f s.', remaining_mission_time)
                else: 
                    rospy.loginfo_throttle(0.5, 'Remaining mission time: %.2f s.', remaining_mission_time)

                if time_from_init > self.time_stop and time_from_init < self.time_start_2:
                    rospy.loginfo('Deactivating robot motion.')
                    self.call_control_activation_service(False)

             
        self.is_robot_static = True

    def call_control_activation_service(self, data: bool):
        msg = SetBoolRequest(data)

        try:
            res = self.srv_cl_activate_control.call(msg)
        except rospy.ServiceException as e:
            rospy.loginfo('Call for activating robot motion failed. Exception caught.')
            return
        
        if res: 
            if data: 
                if not res.success:
                    rospy.loginfo('Call for activating robot motion failed (success = false).')
                else: 
                    rospy.loginfo('Call for activating robot motion was successful.')
                    self.activated = True
            else:
                if not res.success:
                    rospy.loginfo('Call for deactivating robot motion failed (success = false).')
                else: 
                    rospy.loginfo('Call for deactivating robot motion was successful.')
                    self.activated = False


    def cb_odom(self, msg: Odometry): 
        """
        Robot's ground truth odometry callback.

        :param msg: obtained message with ground truth state of the robot
        """

        rospy.loginfo_once('Odometry callback entered')

        if not self.initialized: 
            return

        self.update_traveled_distance(msg)
        self.update_max_dist_from_start(msg)

    def cb_sectors(self, msg: SectorDistances): 
        """
        Callback for distance sectors.

        :param msg: obtained message with minimum distance in particular sectors
        """
        rospy.loginfo_once('Subscribed sector distances.')

        self.pub_dist_left.publish(Float32(msg.distance_left))
        self.pub_dist_front.publish(Float32(msg.distance_front))
        self.pub_dist_right.publish(Float32(msg.distance_right))

    def update_traveled_distance(self, msg: Odometry):
        """
        Updates traveled distance using recent information about robot's position.

        :param msg: obtained message with recent state of the robot
        """

        if not self.prev_odom:
            self.prev_odom = msg
        else: 
            dist_increment = np.sqrt((self.prev_odom.pose.pose.position.x - msg.pose.pose.position.x)**2 + (self.prev_odom.pose.pose.position.y - msg.pose.pose.position.y)**2)
            self.traveled_distance += dist_increment
            
            if dist_increment / (msg.header.stamp - self.prev_odom.header.stamp).to_sec() > self.static_dist_threshold or abs(msg.twist.twist.angular.z) > self.angular_rate_static_threshold:
                self.is_robot_static = False

            self.prev_odom = msg

        self.pub_traveled_distance.publish(Float32(self.traveled_distance))


    def update_max_dist_from_start(self, msg: Odometry):
        """
        Updates maximum distance from start using recent information about robot's position.

        :param msg: obtained message with recent state of the robot
        """
        if not self.init_pose:
            self.init_pose = msg.pose.pose.position
        
        dist_from_start = np.sqrt((self.init_pose.x - msg.pose.pose.position.x)**2 + (self.init_pose.y - msg.pose.pose.position.y)**2)

        self.max_dist_from_start = max(self.max_dist_from_start, dist_from_start)

        self.pub_max_distance_from_start.publish(Float32(self.max_dist_from_start))

    def cb_evaluate_mission(self, req: TriggerRequest) -> TriggerResponse: 
        """
        Callback for aveluate mission service.

        :param req: obtained ROS service request 

        :return: ROS service response 
        """
        rospy.loginfo_once('Mission evaluation callback entered.')

        msg = ""
        if self.mission_evaluation_required: 
            msg = "Evaluation already started"
        else:
            self.time_of_evaluation_request = rospy.get_time()
            msg = "Starting evaluation"

        self.mission_evaluation_required = True

        return TriggerResponse(True, msg)


if __name__ == '__main__':
    rce = ReactiveControllerEvaluator()
    rospy.spin()
