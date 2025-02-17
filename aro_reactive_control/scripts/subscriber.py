#!/usr/bin/env python
import rospy
from std_msgs.msg import String

def callback(msg):
    rospy.loginfo("Received message: %s", msg.data)
    
def msg_listener():

    rospy.init_node('msg_listener', anonymous=True)

    rospy.loginfo('Initializing message listener.')

    rospy.Subscriber("message", String, callback)

    rospy.loginfo('Node initialized. Waiting for messages.')

    rospy.spin()

if __name__ == '__main__':
    msg_listener()
