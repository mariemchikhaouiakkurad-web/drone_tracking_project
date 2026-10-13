import os
from glob import glob

from setuptools import setup

package_name = "drone_target_tracker"

setup(
    name=package_name,
    version="0.1.0",
    # only the node's code is installed; tracker_lib/ (the trackers, ~2 GB with checkpoints) stays in the source tree
    # and is found through the tracker_lib parameter
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "launch"), glob("launch/*.launch.py")),
        (os.path.join("share", package_name, "config"), glob("config/*.yaml")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Ahmed Zghibi",
    maintainer_email="zghibiahmed141@gmail.com",
    description="Single-object tracker node checked by YOLO detections",
    license="MIT",
    tests_require=["pytest"],
    entry_points={"console_scripts": ["tracker_node = drone_target_tracker.tracker_node:main"]},
)
