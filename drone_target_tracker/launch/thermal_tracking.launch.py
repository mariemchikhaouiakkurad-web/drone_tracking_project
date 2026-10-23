"""Camera + detector + tracker in one command.

    ros2 launch drone_target_tracker thermal_tracking.launch.py
    ros2 launch drone_target_tracker thermal_tracking.launch.py tracker:=sutrack detector_every:=10

Starts the same nodes and settings as the workspace's `doc` / yolov8_tensor_rt.launch.py command (camera, DNN image
encoder, TensorRT, YOLOv8 decoder) plus tracker_node, with two differences, without touching the Isaac ROS files:
  * the detector runs every `detector_every` images (default: config/tracker.yaml, 5): the encoder reads the images
    tracker_node forwards (/image_for_detector, /camera_info_for_detector) instead of every camera image;
    detector_every:=0 restores the detector on every image (/image);
  * the decoder gets num_classes 4 (the model's car, bus, truck, motorcycle; its default 80 is wrong for this model).
tracker:=none starts YOLO alone (camera topic -> encoder -> TensorRT -> decoder, no tracker_node): the encoder reads
every /image, as with detector_every:=0. Used to measure the detector by itself.
Paths are the ones inside the Isaac ROS container (/workspaces/isaac_ros-dev/src/...).
"""
import os

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import ComposableNodeContainer, Node
from launch_ros.descriptions import ComposableNode

SRC = "/workspaces/isaac_ros-dev/src"
PKG = get_package_share_directory("drone_target_tracker")


def yaml_param(path, name, default):
    try:
        with open(path) as f:
            return yaml.safe_load(f)["tracker_node"]["ros__parameters"].get(name, default)
    except Exception:
        return default


def setup(context):
    get = lambda name: LaunchConfiguration(name).perform(context)  # noqa: E731
    params_file = get("params_file")
    every = get("detector_every")
    every = int(every) if every != "" else int(yaml_param(params_file, "detector_every", 5))
    yolo_only = get("tracker") == "none"
    if yolo_only:
        every = 0  # no tracker to forward images: the detector reads every camera image
    if every > 0:
        image_in = yaml_param(params_file, "detector_image_topic", "/image_for_detector")
        info_in = yaml_param(params_file, "detector_camera_info_topic", "/camera_info_for_detector")
    else:
        image_in, info_in = "/image", "/camera_info"

    camera = Node(
        package="v4l2_camera", executable="v4l2_camera_node", name="v4l2_camera", output="screen",
        parameters=[{"video_device": get("video_device"), "image_size": [640, 512],
                     "camera_info_url": get("camera_info_url")}],
        remappings=[("image_raw", "image")],
    )
    tensor_rt = ComposableNode(
        name="tensor_rt", package="isaac_ros_tensor_rt", plugin="nvidia::isaac_ros::dnn_inference::TensorRTNode",
        parameters=[{
            "model_file_path": get("model_file_path"),
            "engine_file_path": get("engine_file_path"),
            "input_tensor_names": ["input_tensor"],
            "input_binding_names": ["images"],
            "output_tensor_names": ["output_tensor"],
            "output_binding_names": ["output0"],
            "verbose": False,
            "force_engine_update": False,
        }],
    )
    decoder = ComposableNode(
        name="yolov8_decoder_node", package="isaac_ros_yolov8", plugin="nvidia::isaac_ros::yolov8::YoloV8DecoderNode",
        parameters=[{
            "tensor_name": "output_tensor",
            "confidence_threshold": float(get("confidence_threshold")),
            "nms_threshold": float(get("nms_threshold")),
            "num_classes": 4,
        }],
    )
    container = ComposableNodeContainer(
        name="tensor_rt_container", namespace="", package="rclcpp_components", executable="component_container_mt",
        composable_node_descriptions=[tensor_rt, decoder], output="screen",
    )
    encoder = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(
            get_package_share_directory("isaac_ros_dnn_image_encoder"), "launch", "dnn_image_encoder.launch.py")),
        launch_arguments={
            "input_image_width": "640", "input_image_height": "512",
            "network_image_width": "640", "network_image_height": "512",
            "image_mean": "[0.0, 0.0, 0.0]", "image_stddev": "[1.0, 1.0, 1.0]",
            "attach_to_shared_component_container": "True",
            "component_container_name": "tensor_rt_container",
            "dnn_image_encoder_namespace": "yolov8_encoder",
            "image_input_topic": image_in,
            "camera_info_input_topic": info_in,
            "tensor_output_topic": "/tensor_pub",
        }.items(),
    )
    tracker = Node(
        package="drone_target_tracker", executable="tracker_node", name="tracker_node", output="screen",
        parameters=[params_file, {"tracker": get("tracker"), "detector_every": every}],
    )
    return [camera, container, encoder] if yolo_only else [camera, container, encoder, tracker]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("tracker", default_value="dyhit", description="dyhit, hit_base, sutrack, or none (YOLO alone)"),
        DeclareLaunchArgument("detector_every", default_value="",
                              description="detector every N images; empty = config/tracker.yaml (5); 0 = every image"),
        DeclareLaunchArgument("params_file", default_value=os.path.join(PKG, "config", "tracker.yaml")),
        DeclareLaunchArgument("video_device", default_value="/dev/video0"),
        DeclareLaunchArgument("camera_info_url", default_value=f"file://{SRC}/cameraconfig/thermal_camera.yaml"),
        DeclareLaunchArgument("model_file_path", default_value=f"{SRC}/model/bestperson.onnx"),
        DeclareLaunchArgument("engine_file_path", default_value=f"{SRC}/model/bestperson.plan"),
        DeclareLaunchArgument("confidence_threshold", default_value="0.25"),
        DeclareLaunchArgument("nms_threshold", default_value="0.45"),
        OpaqueFunction(function=setup),
    ])
