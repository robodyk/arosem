#!/usr/bin/env python3
"""
Script for local evaluation of ARO HW 1 - scan processing.
"""
import rospy
import numpy as np
import os
from geometry_msgs.msg import Twist, Vector3
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float32
from std_srvs.srv import Trigger, SetBool, SetBoolRequest, TriggerResponse
from nav_msgs.msg import Odometry
from aro_msgs.msg import SectorDistances

np.set_printoptions(precision=3)

class ScanProcessingEvaluator():

    def __init__(self):
        rospy.loginfo('Initializing node')
        rospy.init_node('scan_processing_evaluator')
        self.initialized = False

        self.sub_sector_dists = rospy.Subscriber('/reactive_control/sector_dists', SectorDistances, self.cb_sector_distances)

        self.ref_solution_file  = rospy.get_param('~ref_solution_file', 'ref_solution.txt')

        self.save_results = rospy.get_param('~save_results', False)
        self.results_file = rospy.get_param('~results_file', 'data/results.tsxt')

        self.srv_evaluate_mission = rospy.Service('/reactive_control/evaluate_mission', Trigger, self.empty_cb)

        self.dists_left_ref = []
        self.dists_right_ref = []
        self.dists_front_ref = []
        self.load_reference_solution(self.ref_solution_file)

        self.dists_left = []
        self.dists_right = []
        self.dists_front = []

        self.last_msg_time = None
        self.sectors_msg_timeout = 2.0
        self.evaluation_timeout = 25.0

        timer_period = 0.5
        self.timer = rospy.Timer(rospy.Duration.from_sec(timer_period), self.timer_cb)

        self.init_time = rospy.get_time()
        self.initialized = True
        rospy.loginfo('Scan processing evaluator initialized.')

    def empty_cb(self, msg):
        """
        Empty callback for a msg. Placeholder for maintaining compatibility across ARO HW1 and HW2.

        :param msg: obtained message
        """
        rospy.loginfo_once('Empty callback entered.')

    def cb_sector_distances(self, msg: SectorDistances):
        """
        Callback for distance sectors.

        :param msg: obtained message with minimum distance in particular sectors
        """
        if not self.initialized: 
            rospy.logerr('First message arrived before initialization was completed.')
            return

        rospy.loginfo_once('Subscribed sector dists')
        self.last_msg_time = rospy.get_time()
        self.dists_left.append(msg.distance_left)
        self.dists_front.append(msg.distance_front)
        self.dists_right.append(msg.distance_right)

    def timer_cb(self, event): 
        """
        Callback function for timer handling checking of elapsed time and initiating evaluation.

        :param event (rospy TimerEvent): event handled by the callback 
        """
        if not self.initialized:
            return

        if self.last_msg_time is None:

            if (rospy.get_time() - self.init_time) > self.evaluation_timeout:
                self.evaluate()
                self.timer.shutdown()

            return

        if (rospy.get_time() - self.last_msg_time) > self.sectors_msg_timeout:
            self.evaluate()
            self.timer.shutdown()
            rospy.signal_shutdown("Evaluation finished")

    def load_reference_solution(self, file_path: str):
        """
        Load reference solution for evaluation from a path.

        :param file_path: path to file containing reference solution 
        """
        rospy.loginfo('Loading reference solution.')
        with open(file_path) as fp:
            for line in fp:
                dist_left, dist_front, dist_right = line.split()
                self.dists_left_ref.append(float(dist_left))
                self.dists_front_ref.append(float(dist_front))
                self.dists_right_ref.append(float(dist_right))
        
        rospy.loginfo('Reference solution of length %lu loaded.', len(self.dists_left_ref))
         

    def evaluate(self) -> bool: 
        """
        Function handling the evaluation of results.

        :return: true if correct data were obtained, false otherwise
        """
        rospy.loginfo('Starting the evalution.')
        success = True

        if self.save_results:
            rospy.loginfo('Saving evaluation data to %s.', self.results_file)
            f = open(self.results_file, "w")
        
        rospy.loginfo('RESULTS:')
        lines = []

        # compare the length of gathered data and reference solution
        if len(self.dists_left) == 0:
            lines.append('No data obtained on topic /reactive_control/sector_dists.')
            rospy.logwarn(lines[-1])
            success = False
        elif len(self.dists_left) != len(self.dists_front) or len(self.dists_left) != len(self.dists_right):
            lines.append('Gathered data for particular sectors do not have an equal length!')
            rospy.logwarn(lines[-1])
            success = False
        elif len(self.dists_left) != len(self.dists_left_ref): 
            lines.append('The length of gathered data {:d} does not equal length of reference solution {:d}!'.format(len(self.dists_left), len(self.dists_left_ref)))
            rospy.logwarn(lines[-1])
            success = False
        else: 
            dist_allowed_deviation = 0.01
            n_violations_left = 0
            n_violations_front = 0
            n_violations_right = 0
            cumulative_dev_left = 0.0
            cumulative_dev_front = 0.0
            cumulative_dev_right = 0.0

            for k in range(len(self.dists_left)):
                deviation_left = abs(self.dists_left[k] - self.dists_left_ref[k])
                deviation_front = abs(self.dists_front[k] - self.dists_front_ref[k])
                deviation_right = abs(self.dists_right[k] - self.dists_right_ref[k])
                if deviation_left > dist_allowed_deviation: 
                    n_violations_left += 1
                    cumulative_dev_left += deviation_left

                if deviation_front > dist_allowed_deviation: 
                    n_violations_front += 1
                    cumulative_dev_front += deviation_front

                if deviation_right > dist_allowed_deviation: 
                    n_violations_right += 1
                    cumulative_dev_right += deviation_right

            if n_violations_left + n_violations_front + n_violations_right > 0: 
                lines.append('Gathered distances in particular sectors do not match reference solution!')
                rospy.logwarn(lines[-1])
                success = False

            lines.append('Number of processed scans: {:d}'.format(len(self.dists_left)))
            rospy.loginfo(lines[-1])   
            lines.append('Number of incorrect data in particular sectors: left = {:d}, front = {:d}, right = {:d}'.format(n_violations_left, n_violations_front, n_violations_right))
            rospy.loginfo(lines[-1])   
            avg_error_left = cumulative_dev_left / float(len(self.dists_left)) 
            avg_error_front = cumulative_dev_front / float(len(self.dists_front)) 
            avg_error_right = cumulative_dev_right / float(len(self.dists_right)) 
            lines.append('Average error in particular sectors: left = {:.2f} m, front = {:.2f} m, right = {:.2f} m'.format(avg_error_left, avg_error_front, avg_error_right))   
            rospy.loginfo(lines[-1])

            avg_error_incorrect_left = 0.0
            avg_error_incorrect_front = 0.0
            avg_error_incorrect_right = 0.0

            if n_violations_left > 0: 
                avg_error_incorrect_left = cumulative_dev_left / float(n_violations_left)

            if n_violations_front > 0: 
                avg_error_incorrect_front = cumulative_dev_front / float(n_violations_front)

            if n_violations_right > 0: 
                avg_error_incorrect_right = cumulative_dev_right / float(n_violations_right)

            lines.append('Average error for incorrect data in particular sectors: left = {:.2f} m, front = {:.2f} m, right = {:.2f} m'.format(avg_error_incorrect_left, avg_error_incorrect_front, avg_error_incorrect_right))   
            rospy.loginfo(lines[-1])

        lines.append('Evaluation result: {0}'.format('Passed' if success else 'Failed'))
        rospy.loginfo(lines[-1])

        if self.save_results:
            for line in lines:
                f.write(line + '\n')
            
            f.close()
            rospy.loginfo('Results saved to %s.', self.results_file)

        return success

if __name__ == '__main__':
    rce = ScanProcessingEvaluator()
    rospy.spin()
