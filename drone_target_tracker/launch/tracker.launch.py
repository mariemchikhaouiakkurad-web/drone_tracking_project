"""Starts tracker_node only (the camera and the detector are started separately, as today).

    ros2 launch drone_target_tracker tracker.launch.py                      # DyHiT
    ros2 launch drone_target_tracker tracker.launch.py tracker:=sutrack     # or hit_base
    ros2 launch drone_target_tracker tracker.launch.py detector_every:=10   # default: config/tracker.yaml (5)
    ros2 launch drone_target_tracker tracker.launch.py params_file:=/path/to/my_tracker.yaml

With detector_every > 0 the node only uses the detections of the images it forwards on /image_for_detector. A
detector started with the workspace's own launch still reads /image and runs on every image (no GPU saved, but the
node behaves the same); launch/thermal_tracking.launch.py starts a detector that reads the forwarded images.
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def setup(context):
    get = lambda name: LaunchConfiguration(name).perform(context)  # noqa: E731
    overrides = {"tracker": get("tracker")}
    if get("detector_every") != "":
        overrides["detector_every"] = int(get("detector_every"))
    return [Node(package="drone_target_tracker", executable="tracker_node", name="tracker_node", output="screen",
                 parameters=[get("params_file"), overrides])]


def generate_launch_description():
    default_params = os.path.join(get_package_share_directory("drone_target_tracker"), "config", "tracker.yaml")
    return LaunchDescription([
        DeclareLaunchArgument("tracker", default_value="dyhit", description="dyhit, hit_base or sutrack"),
        DeclareLaunchArgument("detector_every", default_value="",
                              description="detector every N images; empty = the parameter file's value"),
        DeclareLaunchArgument("params_file", default_value=default_params, description="parameter file"),
        OpaqueFunction(function=setup),
    ])
