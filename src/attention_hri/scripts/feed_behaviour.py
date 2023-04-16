#!/usr/bin/env python3
from __future__ import division
from __future__ import print_function

# import roslib
# roslib.load_manifest('attention_based_hri')
import sys
from collections import deque
import rospy
import cv2
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
from std_msgs.msg import String
from std_msgs.msg import Int32MultiArray

from datetime import datetime
import numpy as np

"""
Preface
  This class subscribes to attention map topic, /Attention/moving_object topic in this case, and 
  provides the center of attention of the saliency map. It is a source of the main behaviour class that directs the 
  interaction between the robot and the environment. 
  
What to modify? 
  if you're working on python 2.x, you may have to underwent extensive library related syntax modifications. However, if
  you plan to work on different environment, the only thing you have to change would be the source of your subscription.
  It is /Attention/moving_object in our case. 

Control flow
  The program looks for published attention map in the aforementioned topic. if any, it segments the ROI, and calculates
  the center of mass of the resulting frame. Then it publishes this center of ROI in Int32MultiArray.
"""

class feed_behaviour():
    def __init__(self):
        rospy.init_node('feed_behavior')

        self.reflux_locomotion = rospy.Publisher("/cmd_vel",Twist, queue_size=5)

        self.attention = rospy.Publisher("attention/center_of_attention",Int32MultiArray, queue_size=10)
        self.bridge = CvBridge()
        # self.saliency_sub = self.create_subscription(Image, "/Attention/salinecy", self.moving_object_callback, 10)
        self.object_sub = rospy.Subscriber("/Attention/attention_image_raw",Image, self.moving_object_model_callback, queue_size=10)
        self.timer = datetime.now()

        # Create Twist message for activator
        self.msg = Twist()
        self.object_sub
        print("Image Processor/Feeder Initiated")

    # callback for center of focuse calculator
    def moving_object_model_callback(self, data): #128X160
        # print("Inside moving object call back")
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)
            print("There was error accepting image")

        # Post process image
        self.post_prediction_process(cv_image)

    # here we draw contours on ROI, calculate the center of ROI and publish center in Int32Array format
    def post_prediction_process(self,image_data):
        # Convert image to grayscale
        gray = cv2.cvtColor(image_data, cv2.COLOR_BGR2GRAY)

        # Apply thresholding to convert image to binary
        ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

        # Find contours in the binary image
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # Calculate area of each contour and append to list
        areas = []
        for contour in contours:
            area = cv2.contourArea(contour)
            areas.append(area)

            # Draw contour on original image
            cv2.drawContours(image_data, [contour], 0, (0, 255, 0), 2)


        # Assumption: Salient or moving object with largest size is most likely to attract the robot.
        # Get index of largest contour
        if(len(areas)>0):
            largest_contour_index = areas.index(max(areas))
            # Calculate center location of largest contour
            moments = cv2.moments(contours[largest_contour_index])
            center_x = int(moments['m10'] / moments['m00'])
            center_y = int(moments['m01'] / moments['m00'])
            center_location = (center_x, center_y)

            int_array_msg = Int32MultiArray()
            int_array_msg.data = center_location

            radius = 5  # radius of circle

            # Draw the circle on the image with a big dot
            color = (0, 0, 255)  # color of the circle (in BGR format)
            thickness = -1  # fill the circle with the specified color
            cv2.circle(image_data, center_location, radius, color, thickness)

            self.activator(1)

            # call activator with value 1
            print("center location: ", int_array_msg)
            self.attention.publish(int_array_msg)
        else:
            # call activator with value 0, when there are no ROI detected
            self.activator(0)

        # Image with contours
        cv2.imshow("Processed Frame", image_data)
        cv2.waitKey(3)

    def activator(self, data):
        if data == 1:
            self.timer = datetime.now()
        else:
            # print(abs(self.timer - datetime.now()))
            if abs(self.timer - datetime.now()).total_seconds() > 5:
                print(abs(self.timer - datetime.now()).total_seconds())

                self.msg.linear.x = 0.25
                self.msg.angular.z = 1.0

                self.reflux_locomotion.publish(self.msg)





def main(args=None):

  feed_behaviour()
  # add exception later
  rospy.spin()

  cv2.destroyAllWindows()

# entry point
if __name__ == '__main__':
    main()