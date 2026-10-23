from glob import glob

from setuptools import setup

package_name = "dalbd_fusion"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml", "README.md"]),
        ("share/" + package_name + "/config", glob("config/*.yaml")),
        ("share/" + package_name + "/launch", glob("launch/*.launch.py")),
        ("share/" + package_name + "/rviz", glob("rviz/*.rviz")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Mariem",
    maintainer_email="mariemchikhaouiakkurad@gmail.com",
    description="DALBD sound / vision fusion state machine",
    license="Proprietary",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "state_machine = dalbd_fusion.state_machine_node:main",
            "fake_sound = dalbd_fusion.fake_sound_node:main",
            "state_report = dalbd_fusion.state_report:main",
            "fusion_viz = dalbd_fusion.fusion_viz_node:main",
            "latency_trace = dalbd_fusion.latency_trace:main",
            "detector_sweep = dalbd_fusion.detector_sweep:main",
            "run_recorder = dalbd_fusion.run_recorder:main",
            "topic_bridge = dalbd_fusion.topic_bridge:main",
        ],
    },
)
