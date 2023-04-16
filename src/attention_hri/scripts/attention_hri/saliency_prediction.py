#!/usr/bin/env python3
from __future__ import division
from __future__ import print_function

# import roslib
# roslib.load_manifest('attention_based_hri')
import sys
from collections import deque
import rospy

import cv2

from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
from std_msgs.msg import String
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
import tensorflow as tf

from attention_hri.saliency_model import *
from attention_hri.config import * # since config is imported in utility fully
from attention_hri.utility import *
# from model import *
# from utility import *
import os
import rospkg
import numpy as np

"""
Preface
  This class subscribes to any live video stream, /CameraTop/image_raw topic in this case, and 
  provides an attention attention map which is empowered by saliency prediction model. Hence, the input for this class 
  would a 2D image published by the robot either in real of simulated environment. The output of this class would be an 
  attention map, a result of video saliency prediction model. 
  
What to modify? 
  if you're working on python 2.x, you may have to underwent extensive library related syntax modifications. However, if
  you plan to work on different environment, the only thing you have to change would be the source of your subscription.
  It is /CameraTop.image_raw in our case. 

Control flow
  The program looks for published frames in the aforementioned topic. if any, it passes it to both of our attention 
  function, namely - saliency prediction model function. 
  ---
  This module publishes to/Attention/attention_image_raw.
"""

class saliency_prediction():
  
  def __init__(self):
    rospy.init_node('saliency_prediction')
    self.saliency_map_pub = rospy.Publisher("/Attention/attention_image_raw",Image, queue_size=10)
    # self.localize_pub = self.create_publisher(String,"attention_based_hri/object_roi", 10)

    self.bridge = CvBridge()
    # self.image_sub = self.create_subscription(Image,"/naoqi_driver/camera/front/image_raw", self.saliency_prediction_callback,10)
    self.image_sub = rospy.Subscriber("/naoqi_driver/camera/front/image_raw", Image,self.saliency_prediction_callback, queue_size=10)

    self.image_sub

    self.x = Input(batch_shape=(1, None, shape_r, shape_c, 3))
    # self.x2 = Input(batch_shape=(1, None, shape_r, shape_c, 3))
    # self.x3 = Input(batch_shape=(1, None, shape_r, shape_c, 3))
    self.stateful = True

    rp = rospkg.RosPack()
    package_path = rp.get_path('attention_hri')

    self.m = Model(inputs=self.x, outputs=saliency_model_entry(self.x, self.stateful))
    self.m.load_weights(os.path.join(package_path, 'resource/ivsp.h5')) # please change this relative path according to your finle orgnization

    self.queue = deque()
    print("Attention Module - Saliency Prediction - Initiated")

  # callback for saliency prediction component
  def saliency_prediction_callback(self, data):
    try:

      cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
    except CvBridgeError as e:
      print(e)

    cv2.imshow("Center of Focus", cv_image)
    cv2.waitKey(3)
   
    # get access to video
    # this section pops oldest and push newest for every frame published -> =|=|=| ->
    if len(self.queue) != num_frames:
      self.queue.append(cv_image)
    else:
      self.queue.popleft()
      self.queue.append(cv_image)

      # the length of the frame is tantamaout to the size of our batch
      # print(len(self.queue))

      Xims = np.zeros((1, len(self.queue), shape_r, shape_c, 3))
      # Xims2 = np.zeros((1, len(self.queue), shape_r, shape_c, 3))
      # Xims3 = np.zeros((1, len(self.queue), shape_r, shape_c, 3))

      [X, X2, X3] = preprocess_images_realtime(self.queue, shape_r, shape_c)

      # print(X.shape, "X shape new")
      Xims[0] = np.copy(X)
      # Xims2[0] = np.copy(X2)
      # Xims3[0] = np.copy(X3)

      prediction = self.m.predict(Xims)
      print("Prediction shape: ", prediction.shape)

      for j in range(len(self.queue)):
        orignal_image = self.queue[0]

        # print(orignal_image.shape, "Queue shape")

        x, y = divmod(j, len(self.queue))
        # print(x, y)


        # cv.imshow("Frame", prediction[0,0,:,:,0] )
        # cv.waitKey(3)

      print(prediction[0,0,:,:,0].shape)


      # predictor is called here
      self.publish_predicted(prediction[0,0,:,:,0])

      self.m.reset_states()

        
  
  """ Saliency Prediction Model """
  def publish_predicted(self, image_data):
    # print("Publishable data received")

    # here converting image format to bgr8
    img_bgr = cv2.cvtColor(image_data, cv2.COLOR_GRAY2BGR)
    predicted_map = np.uint8(img_bgr * 255)
    predicted_map = scale_image(predicted_map)

    # cv2.imshow("Image from Robot", predicted_map)
    # cv2.waitKey(3)

    # Publishing attention map
    # add exception if needed?
    self.saliency_map_pub.publish(self.bridge.cv2_to_imgmsg(predicted_map, "bgr8"))

    # print(predicted_map.shape, "Data published")

# Original image with size 128X160 not good for pixel distance to Twist msg type conversion. Scale X4
def scale_image(img):
  """Returns the input image with double the size"""
  height, width = img.shape[:2]
  new_height, new_width = height * 4, width * 4
  return cv2.resize(img, (new_width, new_height))




def main(args=None):
  saliency_prediction()
  # add exception later
  rospy.spin()

  cv2.destroyAllWindows()

# entry point
if __name__ == '__main__':
    main()

