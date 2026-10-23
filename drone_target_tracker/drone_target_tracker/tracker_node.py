"""
tracker_node.py - ROS 2 node that follows ONE target with a single-object tracker (SUTrack, HiT-Base or DyHiT),
checked and corrected by the YOLO detections of isaac_ros_yolov8.

Subscribes   /image               sensor_msgs/Image                 every camera frame -> the tracker
             /detections_output   vision_msgs/Detection2DArray      the detector's boxes (same timestamps as /image)
             /camera_info         sensor_msgs/CameraInfo            focal length and centre, for the bearing
Publishes    /target              vision_msgs/Detection2D           the target's box, on every image while tracking
             /target/bearing      geometry_msgs/Vector3Stamped      x = azimuth, y = elevation (radians; right / up
                                                                    positive), from the box centre and /camera_info
             /target/status       std_msgs/String                   "tracking" or "searching"
             /target/debug_image  sensor_msgs/Image (bgr8)          the image with the boxes (publish_debug_image)
             /image_for_detector  sensor_msgs/Image                 the images the detector should process: every
             /camera_info_for_detector  sensor_msgs/CameraInfo      N-th image while tracking (detector_every, 5),
                                                                    every image while searching; the detector's
                                                                    encoder must read these two topics
                                                                    (launch/thermal_tracking.launch.py does it)
The decisions (confirm / correct / switch / lost / search) are in logic.py, the image-detection matching in core.py.
"""
import math
import time

import rclpy
from geometry_msgs.msg import Vector3Stamped
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import String
from vision_msgs.msg import Detection2D, Detection2DArray, ObjectHypothesisWithPose

from drone_target_tracker.core import TrackerCore, load_tracker
from drone_target_tracker.image_conv import bgr_fields, to_bgr
from drone_target_tracker.logic import Detection, TargetLogic

RED, CYAN, GREEN, YELLOW, WHITE = (0, 0, 255), (255, 255, 0), (0, 200, 0), (0, 220, 255), (255, 255, 255)


def stamp_s(header):
    return header.stamp.sec + header.stamp.nanosec * 1e-9


class TrackerNode(Node):
    def __init__(self):
        super().__init__("tracker_node")
        p = self.declare_parameter
        self.tracker_name = p("tracker", "dyhit").value
        tracker_lib = p("tracker_lib", "/workspaces/isaac_ros-dev/src/drone_target_tracker/tracker_lib").value
        image_topic = p("image_topic", "/image").value
        detections_topic = p("detections_topic", "/detections_output").value
        camera_info_topic = p("camera_info_topic", "/camera_info").value
        self.class_names = list(p("class_names", ["car", "bus", "truck", "motorcycle"]).value)
        self.target_classes = set(p("target_classes", ["car"]).value)
        self.min_score = float(p("min_score", 0.25).value)
        verify_iou = p("verify_iou", 0.3).value
        reacquire_gate = p("reacquire_gate", 1.5).value
        lost_after = p("lost_after", 3).value
        lost_after_s = float(p("lost_after_s", 0.5).value)
        search_radius = p("search_radius", 0.0).value
        search_growth = float(p("search_growth", 200.0).value)
        trust_tracker_s = float(p("trust_tracker_s", 2.0).value)
        initial_target = p("initial_target", "best").value
        max_catch_up = p("max_catch_up", 3).value
        stamp_tolerance = p("stamp_tolerance", 0.02).value
        lost_score_param = float(p("lost_score", -1.0).value)
        self.detector_every = int(p("detector_every", 5).value)
        detector_image_topic = p("detector_image_topic", "/image_for_detector").value
        detector_info_topic = p("detector_camera_info_topic", "/camera_info_for_detector").value
        check_on_low_score = bool(p("check_on_low_score", True).value)
        self.publish_debug = bool(p("publish_debug_image", True).value)
        log_every = float(p("log_every_s", 5.0).value)

        self.get_logger().info(f"loading tracker '{self.tracker_name}' from {tracker_lib} (can take a minute) ...")
        t0 = time.perf_counter()
        tracker, lost_score = load_tracker(tracker_lib, self.tracker_name)
        if lost_score_param >= 0:
            lost_score = lost_score_param
        self.get_logger().info(f"tracker '{self.tracker_name}' ready on {getattr(tracker, 'device', 'cpu')} in "
                               f"{time.perf_counter() - t0:.1f} s (lost_score {lost_score})")
        logic = TargetLogic(verify_iou, reacquire_gate, lost_after, search_radius, initial_target,
                            lost_after_s=lost_after_s, search_growth=search_growth,
                            trust_tracker_s=trust_tracker_s)
        self.core = TrackerCore(tracker, logic, lost_score, max_catch_up, stamp_tolerance,
                                detector_every=self.detector_every, check_on_low_score=check_on_low_score)

        self.K = None  # (fx, fy, cx, cy) from /camera_info
        self.camera_info = None  # the latest CameraInfo, forwarded with the images sent to the detector
        self.last_dets = (None, [])  # (stamp, detections) for the debug image

        latest_only = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST)
        self.create_subscription(Image, image_topic, self.on_image, latest_only)
        self.create_subscription(Detection2DArray, detections_topic, self.on_detections, 10)
        self.create_subscription(CameraInfo, camera_info_topic, self.on_camera_info, qos_profile_sensor_data)
        self.pub_target = self.create_publisher(Detection2D, "/target", 10)
        self.pub_bearing = self.create_publisher(Vector3Stamped, "/target/bearing", 10)
        self.pub_status = self.create_publisher(String, "/target/status", 10)
        self.pub_debug = self.create_publisher(Image, "/target/debug_image", 2) if self.publish_debug else None
        self.pub_det_image = self.pub_det_info = None
        if self.detector_every > 0:
            self.pub_det_image = self.create_publisher(Image, detector_image_topic, 2)
            self.pub_det_info = self.create_publisher(CameraInfo, detector_info_topic, 2)
        if log_every > 0:
            self.create_timer(log_every, self.log_summary)
        self.get_logger().info(
            f"listening to {image_topic} and {detections_topic}; target classes {sorted(self.target_classes)}; "
            + (f"detector every {self.detector_every} images (sent on {detector_image_topic})" if self.detector_every
               else "detector_every 0: every detection message is used"))

    # ------------------------------------------------------------------ callbacks
    def on_camera_info(self, msg):
        self.camera_info = msg
        k = msg.k
        if k[0] > 0 and k[4] > 0:
            self.K = (k[0], k[4], k[2], k[5])

    def on_image(self, msg):
        stamp = stamp_s(msg.header)
        # 1) low latency: decide first and forward the received message as it is, BEFORE converting it and BEFORE
        #    the tracker runs, so YOLO starts ~20 ms earlier (it used to wait for the tracker on every check)
        sent = self.core.pre_image(stamp) if self.pub_det_image is not None else False
        if sent:
            self.forward_to_detector(msg)
        try:
            frame = to_bgr(msg)
        except ValueError as e:
            self.get_logger().error(str(e), throttle_duration_sec=5.0)
            return
        # 2) then the tracker (its result does not depend on the detection of this image, which comes later anyway)
        res = self.core.on_image(frame, stamp, sent=sent)
        if res.send and not sent:  # late check asked by a low tracker score (SUTrack only)
            self.forward_to_detector(msg)
        self.publish_outputs(msg.header, frame, res)

    def forward_to_detector(self, msg):
        if self.camera_info is not None:  # the encoder pairs each image with a CameraInfo of the same stamp
            info = self.camera_info
            info.header.stamp = msg.header.stamp
            self.pub_det_info.publish(info)
        self.pub_det_image.publish(msg)

    def on_detections(self, msg):
        stamp = stamp_s(msg.header)
        dets = self.parse(msg)
        self.last_dets = (stamp, dets)
        dec = self.core.on_detections(dets, stamp)
        if dec is not None and dec.event in ("init", "reacquire", "correct", "lost"):
            self.get_logger().info(f"{dec.event} (stamp {stamp:.3f}, {len(dets)} target-class detections)")

    def parse(self, msg):
        out = []
        for d in msg.detections:
            if not d.results:
                continue
            hyp = d.results[0].hypothesis
            try:
                i = int(hyp.class_id)
                label = self.class_names[i] if 0 <= i < len(self.class_names) else None  # bad ids are dropped
            except ValueError:
                label = hyp.class_id
            if label not in self.target_classes or hyp.score < self.min_score:
                continue
            c = d.bbox.center
            cx, cy = (c.position.x, c.position.y) if hasattr(c, "position") else (c.x, c.y)
            w, h = d.bbox.size_x, d.bbox.size_y
            if w > 0 and h > 0:
                out.append(Detection((cx - w / 2, cy - h / 2, w, h), float(hyp.score), label))
        return out

    # ------------------------------------------------------------------ outputs
    def publish_outputs(self, header, frame, res):
        self.pub_status.publish(String(data="tracking" if res.box is not None else "searching"))
        if res.box is not None:
            x, y, w, h = res.box
            det = Detection2D()
            det.header = header
            det.id = "target"
            c = det.bbox.center
            if hasattr(c, "position"):
                c.position.x, c.position.y = x + w / 2, y + h / 2
            else:
                c.x, c.y = x + w / 2, y + h / 2
            det.bbox.size_x, det.bbox.size_y = float(w), float(h)
            hyp = ObjectHypothesisWithPose()
            label = next(iter(self.target_classes)) if len(self.target_classes) == 1 else "target"
            hyp.hypothesis.class_id = (str(self.class_names.index(label)) if label in self.class_names else label)
            hyp.hypothesis.score = float(res.score) if res.score is not None else 0.0
            det.results.append(hyp)
            self.pub_target.publish(det)
            if self.K is not None:
                fx, fy, cx, cy = self.K
                b = Vector3Stamped()
                b.header = header
                b.vector.x = math.atan2(x + w / 2 - cx, fx)   # azimuth, right positive
                b.vector.y = -math.atan2(y + h / 2 - cy, fy)  # elevation, up positive
                self.pub_bearing.publish(b)
        if self.pub_debug is not None and self.pub_debug.get_subscription_count() > 0:
            self.publish_debug_image(header, frame, res)

    def publish_debug_image(self, header, frame, res):
        import cv2
        img = frame.copy()
        dstamp, dets = self.last_dets
        if dstamp is not None and abs(dstamp - res.stamp) < 0.5:
            for d in dets:
                x, y, w, h = (int(round(v)) for v in d.box)
                cv2.rectangle(img, (x, y), (x + w, y + h), CYAN, 1)
        if res.box is not None:
            x, y, w, h = (int(round(v)) for v in res.box)
            cv2.rectangle(img, (x, y), (x + w, y + h), RED, 2)
        ev = self.core.last_decision[1].event if self.core.last_decision else ""
        c = self.core.logic.counts
        lines = [f"{self.tracker_name}  {'TRACKING' if res.box is not None else 'SEARCHING'}"
                 + (f"  trk {res.trk_ms:.1f} ms" if res.trk_ms is not None else "")
                 + (f"  score {res.score:.2f}" if res.score is not None else ""),
                 f"last check: {ev}   confirmed {c['confirmed']}  corrected {c['corrected']}  "
                 f"switches {c['switches']}  lost {c['lost']}  reacquired {c['reacquired']}"]
        for i, t in enumerate(lines):
            cv2.putText(img, t, (8, 18 + 18 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1, cv2.LINE_AA)
        out = Image(**bgr_fields(img))
        out.header = header
        self.pub_debug.publish(out)

    def log_summary(self):
        s = self.core.summary()
        keys = ["mode", "images", "tracked", "trk_ms_mean", "trk_ms_p95", "sent_to_detector", "detection_msgs",
                "ignored_detections", "stale_detections",
                "checks", "confirmed", "corrected", "switches", "lost", "reacquired", "restarts", "restart_ms_mean"]
        self.get_logger().info("  ".join(f"{k}={s[k]}" for k in keys if k in s))


def main():
    rclpy.init()
    node = TrackerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.log_summary()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
