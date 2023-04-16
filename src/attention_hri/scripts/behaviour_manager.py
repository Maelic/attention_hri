#!/usr/bin/env python3
from __future__ import division
from __future__ import print_function

# import roslib
# roslib.load_manifest('attention_based_hri')
import sys
from collections import deque
from geometry_msgs.msg import Twist
import rospy
import cv2

from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
from std_msgs.msg import String
from std_msgs.msg import Int32MultiArray

from attention_hri.utility import *

"""
Preface
  This is the main behaviour class that subscribes to Int32MultiArray type "attention/center_of_attention topic/center 
  of focus/ and calculates how far the salient object is from the center of the overall frame/environment. Finally, it 
  converts the difference in pixel to geometry_msgs/Twist type robot locomotion parameters like linear and angular 
  parameters. The center of attention of the saliency map. It is a source of the main behaviour class that directs the 
  interaction between the robot and the environment. 

What to modify? 
  if you're working on python 2.x, you may have to underwent extensive library related syntax modifications. However, if
  you plan to work on different environment, the only thing you have to change would be the source of your subscription.

Control flow
  The program checks for center of focus information, it calculates the center of ROI difference from the overall frame 
  center, then it converts the difference pixel wise distance to linear and angular data and publish it to the topic 
  where the robot listens for locomotion. The idea is for the robot to face the center of focus/ROI or the salient object
 directly, making past ROI which might be far from the center of the original frame appear in the center of the 
 environment, which in turn let the robot interact with the ROI eye-to-eye and intuitive manner.
 --- 
 Please make sure to run:
 ros2 control list_hardware_interfaces
 ros2 control list_controllers
 ros2 run controller_manager spawner diff_cont
 
 make the topic the robot listens to /diff_cont/cmd_vel_unstamped and the value of PlanarDriven = True for this code
 to work well. 
 
"""

class behaviour_manager():

    def __init__(self):
        rospy.init_node('behavior_manager')
        self.locomotion = rospy.Publisher("/cmd_vel",Twist, queue_size=10)

        self.bridge = CvBridge()
        self.center_of_focus = rospy.Subscriber("attention/center_of_attention", Int32MultiArray, self.center_of_focus_callback, queue_size=10)

        self.center_of_focus

        # Create Twist message
        self.msg = Twist()
        print("Behaviour Manager Initiated")


    # This function accepts center of couse coordinates, and converts it to equivalent Twist parameter.
    def center_of_focus_callback(self, data):

        # This shouldn't be hard-coded for production environment
        IMAGE_WIDTH = 640
        IMAGE_HEIGHT = 480

        # Extract center location from message
        center_location = data.data

        # Calculate distance from center of image
        distance_x = center_location[0] - IMAGE_WIDTH / 2
        distance_y = center_location[1] - IMAGE_HEIGHT / 2

        # Print distance to console
        print("Center location is {} pixels from image center ({} to the right, {} down)".format(
            (distance_x ** 2 + distance_y ** 2) ** 0.5, distance_x, distance_y))

        # Convert the pixel-wise distance to robot velocity commands
        kp = 0.01  # Proportional gain (tune this value to adjust the speed of movement)
        max_vel = 0.1  # Maximum velocity of the robot

        # still more work for perfect conversion
        #self.msg.linear.x = - max_vel * kp * distance_y
        # self.msg.linear.x = max_vel * kp * distance_x
        self.msg.angular.z = max_vel * kp * distance_x
        self.msg.angular.z = - self.msg.angular.z
        # self.msg.angular.z = max_vel * kp * error_x


        # print(self.msg)
        # exception handling here, incase
        # Publish message
        self.locomotion.publish(self.msg)

    def stop_controller(self):
        rospy.loginfo("Stopping the robot...")
        msg = Twist()
        msg.linear.x = 0
        msg.angular.z = 0
        self.locomotion.publish(msg)

def main(args=None):
  manager = behaviour_manager()
  rospy.on_shutdown(manager.stop_controller)

  rate = rospy.Rate(10) # ROS Rate at 2Hz

  while not rospy.is_shutdown():
      # rospy.loginfo("Hello")
      rate.sleep()

  cv2.destroyAllWindows()

# entry point
if __name__ == '__main__':
    main()
