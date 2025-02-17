#!/usr/bin/env python
import rospy
from std_msgs.msg import String

def msg_publisher():

    rospy.init_node('msg_publisher', anonymous=True)

    rospy.loginfo('Initializing message publisher.')

    pub = rospy.Publisher('message', String, queue_size=10, latch=True)

    rate = rospy.Rate(1)

    counter = 0
    rospy.loginfo('Node initialized. Starting message publishing.')

    while not rospy.is_shutdown():
        pub.publish("hello #{0:d}".format(counter))
        counter = counter + 1
        rate.sleep()

if __name__ == '__main__':
    try:
        msg_publisher()
    except rospy.ROSInterruptException:
        pass
