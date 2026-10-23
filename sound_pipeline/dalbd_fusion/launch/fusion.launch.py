"""fusion.launch.py - starts the DALBD fusion state machine.

  ros2 launch dalbd_fusion fusion.launch.py
  ros2 launch dalbd_fusion fusion.launch.py sound_domain:=42 vision_domain:=0 sound_yaw_offset_deg:=3.5
  ros2 launch dalbd_fusion fusion.launch.py fake_sound:=true schedule:="0-20:10/5,20-30:off,30-60:10/5"
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def setup(context, *args, **kwargs):
    cfg = os.path.join(get_package_share_directory("dalbd_fusion"), "config", "fusion.yaml")
    get = lambda n: LaunchConfiguration(n).perform(context)  # noqa: E731
    vdom, sdom = get("vision_domain"), get("sound_domain")
    overrides = {}
    for name in ("sound_yaw_offset_deg", "sound_pitch_offset_deg", "sound_az_sign"):
        v = get(name)
        if v:
            overrides[name] = float(v)
    if get("sound_requires_loss"):
        overrides["sound_requires_loss"] = get("sound_requires_loss").lower() == "true"
    actions = [Node(
        package="dalbd_fusion", executable="state_machine", output="screen",
        parameters=[cfg, overrides] if overrides else [cfg],
        additional_env={"FUSION_VISION_DOMAIN": vdom, "FUSION_SOUND_DOMAIN": sdom, "ROS_DOMAIN_ID": vdom},
    )]
    if get("fake_sound").lower() == "true":
        actions.append(Node(
            package="dalbd_fusion", executable="fake_sound", name="fake_sound", output="screen",
            parameters=[{"schedule": get("schedule"), "noise_deg": 2.0}],
            additional_env={"ROS_DOMAIN_ID": sdom},
        ))
    return actions


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("vision_domain", default_value="0"),
        DeclareLaunchArgument("sound_domain", default_value="42"),
        DeclareLaunchArgument("sound_yaw_offset_deg", default_value=""),
        DeclareLaunchArgument("sound_pitch_offset_deg", default_value=""),
        DeclareLaunchArgument("sound_az_sign", default_value=""),
        DeclareLaunchArgument("sound_requires_loss", default_value=""),
        DeclareLaunchArgument("fake_sound", default_value="false"),
        DeclareLaunchArgument("schedule", default_value="0-20:10/5,20-30:off,30-60:10/5"),
        OpaqueFunction(function=setup),
    ])
