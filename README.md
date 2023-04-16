# anthropomorphic_hri
This repository corresponds to a research project entitled Anthropomorphic Human Robot Interaction Framework: Attention Based Approach
<p align=""><img src="gitcover.png" width="1000"\></p>

## How does it work
This framework subscribes to any robot camera view topic, /CameraBottom/image_raw  in this case and undergo human attention prediction based on two models namely: saliency prediction and mobing object detection and segmentation. It calulates saliency map and directs the robot to interact with the best ROI according to results. 
## Implementation
* Follow every step in https://github.com/RoboBreizh-RoboCup-Home/gazebo_ros2_pepper and Set PlanarDriven = True
* navigate to this workspace and build it, preferably using colcon build
* use ros2 launch attention_hri hri_attention.py to launch the framework. 
* Make sure gazebo is running well and pepper robot along with the cafe.world are spawned successfully. Also make sure that the topics this module subscribes to are available. mainly, /CameraTop/image_raw. 
*Make sure the robot is listining to /diff_cont/cmd_vel_unstamped for locomotion. Otherwise change the topics in the code accordingly.
## Tools and Platforms
* ROS2 Humble Version
* Gazebo 11.x
* Python 3.9 
* environment is setup to have all tensorflow, keras, opencv, numpy related packages. 
