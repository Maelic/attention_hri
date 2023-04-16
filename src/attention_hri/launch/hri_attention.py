import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, IncludeLaunchDescription
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch.launch_description_sources import PythonLaunchDescriptionSource


def generate_launch_description():

    # main package path
    package_dir=get_package_share_directory('attention_hri')
    return LaunchDescription([
        #   main class
            Node(
                package='attention_hri',
                executable='moving_object',
                name='moving_object',
                output='screen'),

        #  behaviour shaper
            Node(
                package='attention_hri',
                executable='center_calculator',
                name='center_calculator',
                output='screen'),
        #  simple behaviour class
            Node(
                package='attention_hri',
                executable='behaviour',
                name='behaviour',
                output='screen')
    ])
