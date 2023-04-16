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

# attempt to automatically switch backend to easily switch from local to onboard implementation 
try:
  from tensorflow.keras.layers import Input
  from tensorflow.keras.models import Model
  USE_TFLITE=False
except:
  import tflite_runtime.interpreter as tflite
  USE_TFLITE=True

from attention_hri.object_model import *
from attention_hri.config import * # since config is imported in utility fully
from attention_hri.utility import *
# from model import *
# from utility import *
import rospkg
import os
import numpy as np

"""
Preface
  This class subscribes to any live video stream, /CameraTop/image_raw topic in this case, and 
  provides an attention attention map which is empowered by saliency prediction model and moving object detection and 
  segmentation model. Hence, the input for this class would a 2D image published by the robot either in real of simulated
  environment. The output of this class would be an attention map, either a result of moving object detetion or a result 
  of video saliency prediction model. 
  
What to modify? 
  if you're working on python 2.x, you may have to underwent extensive library related syntax modifications. However, if
  you plan to work on different environment, the only thing you have to change would be the source of your subscription.
  It is /CameraTop.image_raw in our case. 

Control flow
  The program looks for published frames in the aforementioned topic. if any, it passes it to both of our attention 
  functions, namely - saliency prediction model function and moving object detection and segmentation function. 
  ---
  This module publishes to /Attention/attention_image_raw, assuming movement and saliency complement with each other 
  most of the time. However, the other option would be to publish in two different topic and handle them in behaviour 
  class for a more sophsticated functionality. 
"""

class moving_object_detection():
  
  def __init__(self, use_tflite=None):
    rospy.init_node('moving_object')

    self.object_map_pub = rospy.Publisher("/Attention/attention_image_raw", Image, queue_size=10)
    # self.localize_pub = self.create_publisher(String,"attention_based_hri/object_roi", 10)

    self.bridge = CvBridge()
    # self.image_sub = self.create_subscription(Image,"/naoqi_driver/camera/front/image_raw", self.moving_object_callback,10)
    self.image_sub = rospy.Subscriber("/naoqi_driver/camera/front/image_raw",Image, self.moving_object_callback, queue_size=10)

    rp = rospkg.RosPack()
    package_path = rp.get_path('attention_hri')
    model_file = os.path.join(package_path, 'resource/XYShift_model.tflite')
    if use_tflite is not None:
      self.use_tflite = use_tflite
    else:
      self.use_tflite = USE_TFLITE

    if self.use_tflite:
      self.m = tflite.Interpreter(model_path=model_file)
    else:
      self.x = Input(batch_shape=(1, None, shape_r, shape_c, 3))
      self.x2 = Input(batch_shape=(1, None, shape_r, shape_c, 3))
      self.x3 = Input(batch_shape=(1, None, shape_r, shape_c, 3))
      self.stateful = True
      self.m = Model(inputs=[self.x, self.x2, self.x3], outputs=transform_saliency([self.x, self.x2, self.x3], self.stateful))
      self.m.load_weights(os.path.join(package_path, 'resource/XYshift.h5')) # please change this relative path according to your finle orgnization
      
    self.queue = deque()

    # converter = tf.lite.TFLiteConverter.from_keras_model(self.m)
    # tflite_model = converter.convert()

    # # Save the model.
    # with open('/home/maelic/Documents/NATNAEL/catkin_ws/src/attention_hri/resource/XYShift.tflite', 'wb') as f:
    #   f.write(tflite_model)

    print("Attention Module - Moving Object Detection Initiated")
    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    # rospy.shutdown()
    
  # callback for saliency prediction component
  def moving_object_callback(self, data):
    try:
      cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
    except CvBridgeError as e:
      print(e)

    # resize image to 640x480 for better output
    img_to_show = cv2.resize(cv_image, (640, 480))
    cv2.imshow("Center of Focus", img_to_show)
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
      Xims2 = np.zeros((1, len(self.queue), shape_r, shape_c, 3))
      Xims3 = np.zeros((1, len(self.queue), shape_r, shape_c, 3))

      [X, X2, X3] = preprocess_images_realtime(self.queue, shape_r, shape_c)

      # print(X.shape, "X shape new")
      Xims[0] = np.copy(X)
      Xims2[0] = np.copy(X2)
      Xims3[0] = np.copy(X3)

      # cast to fp32
      Xims = Xims.astype('float32')
      Xims2 = Xims2.astype('float32')
      Xims3 = Xims3.astype('float32')

      if self.use_tflite:
        self.m.allocate_tensors()

        # Get input and output tensors.
        input_details = self.m.get_input_details()
        output_details = self.m.get_output_details()

        #input_data = [Xims,Xims2,Xims3]
        self.m.resize_tensor_input(input_details[0]['index'], Xims.shape)
        self.m.resize_tensor_input(input_details[1]['index'], Xims2.shape)
        self.m.resize_tensor_input(input_details[2]['index'], Xims3.shape)
        print(input_details[0]['index'])
        self.m.allocate_tensors()

        self.m.set_tensor(input_details[0]['index'], Xims)
        self.m.set_tensor(input_details[1]['index'], Xims2)
        self.m.set_tensor(input_details[2]['index'], Xims3)

        self.m.invoke()

        # The function `get_tensor()` returns a copy of the tensor data.
        # Use `tensor()` in order to get a pointer to the tensor.
        prediction = self.m.get_tensor(output_details[0]['index'])
      else:
        prediction = self.m.predict([Xims,Xims2,Xims3])
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
      self.predict(prediction[0,0,:,:,0])

      if self.use_tflite:
        self.m.reset_all_variables()
      else:
        self.m.reset_states()

        
  
  """ Moving object detection and segmentation model """
  def predict(self, image_data):
    """
    load model here
    predict
    """
    # print("Publishable data received")

   # here converting image format to bgr8
    img_bgr = cv2.cvtColor(image_data, cv2.COLOR_GRAY2BGR)
    predicted_map = np.uint8(img_bgr * 255)
    predicted_map = scale_image(predicted_map)

    # cv2.imshow("Image from Robot", predicted_map)
    # cv2.waitKey(3)

    # Publishing attention map
    # add exception if needed?
    self.object_map_pub.publish(self.bridge.cv2_to_imgmsg(predicted_map, "bgr8"))

    # print(predicted_map.shape, "Data published")

# Original image with size 128X160 not good for pixel distance to Twist msg type conversion. Scale X4
def scale_image(img):
  """Returns the input image with double the size"""
  height, width = img.shape[:2]
  new_height, new_width = height * 4, width * 4
  return cv2.resize(img, (new_width, new_height))


def main():

  moving_object_detection(use_tflite=False)
  # add exception later
  rospy.spin()

  cv2.destroyAllWindows()

# entry point
if __name__ == '__main__':
    main()

