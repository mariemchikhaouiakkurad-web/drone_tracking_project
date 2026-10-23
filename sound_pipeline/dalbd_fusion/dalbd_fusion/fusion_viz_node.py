#!/usr/bin/env python3
"""fusion_viz_node.py - RViz view of the DALBD fusion (run it in the SOUND domain, e.g. ROS_DOMAIN_ID=42,
where /fused/state and /drone_warning both exist; open RViz on the laptop in the same domain).

Publishes /fusion/viz (visualization_msgs/MarkerArray, frame "camera": x forward, y left, z up) and a static TF
camera -> odas built from sound_axes, so the original UMA-8 markers (/target/markers, frame "odas") are drawn
in the camera frame too.

  grey pyramid      camera field of view (UV640-9: 46.2 x 37.7 deg)
  YELLOW arrow      live sound direction from /drone_warning (best DRONE target), + label LEFT/RIGHT UP/DOWN
  grey ball         its mirror point behind the board (planar array: front/back ambiguity)
  yellow dots       sound trail (last ~3 s)
  BLUE / GREEN      fused output direction (/fused/state): blue = VISION, green = SOUND
  big text          current state + source

Parameters: sound_axes "forward=+z,left=-y,up=+x" (same as the fusion) | hfov_deg 46.2 | vfov_deg 37.7
            arrow_len 2.5 | fov_len 3.0 | rate_hz 10.0
Run (Isaac ROS container):  ROS_DOMAIN_ID=42 ros2 run dalbd_fusion fusion_viz
"""
import json
import math
import time

import rclpy
from geometry_msgs.msg import Point, TransformStamped
from rclpy.node import Node
from std_msgs.msg import ColorRGBA, String
from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster
from visualization_msgs.msg import Marker, MarkerArray

from dalbd_fusion.sound_parsing import parse_axes, warning_xyz, xyz_to_camera

GREY, YELLOW, BLUE, GREEN, WHITE, RED = ((0.7, 0.7, 0.7, 0.8), (1.0, 0.85, 0.1, 1.0), (0.25, 0.55, 1.0, 1.0),
                                       (0.2, 0.85, 0.35, 1.0), (1.0, 1.0, 1.0, 1.0), (1.0, 0.3, 0.3, 1.0))


def col(c):
    return ColorRGBA(r=c[0], g=c[1], b=c[2], a=c[3])


def unit(az_right_deg, el_deg):
    """camera az (right +) / el (up +) -> unit vector in ROS camera frame (x fwd, y left, z up)."""
    a, e = math.radians(-az_right_deg), math.radians(el_deg)
    return (math.cos(e) * math.cos(a), math.cos(e) * math.sin(a), math.sin(e))


def words(az, el):
    h = "CENTRE" if abs(az) < 2 else f"{'RIGHT' if az > 0 else 'LEFT'} {abs(az):.0f}deg"
    v = "" if abs(el) < 2 else f" {'UP' if el > 0 else 'DOWN'} {abs(el):.0f}deg"
    return h + v


def quat_from_rows(M):
    """rotation matrix (rows = camera x,y,z expressed in odas axes) -> quaternion (x, y, z, w)."""
    m00, m01, m02 = M[0]
    m10, m11, m12 = M[1]
    m20, m21, m22 = M[2]
    tr = m00 + m11 + m22
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2
        return ((m21 - m12) / s, (m02 - m20) / s, (m10 - m01) / s, 0.25 * s)
    if m00 > m11 and m00 > m22:
        s = math.sqrt(1.0 + m00 - m11 - m22) * 2
        return (0.25 * s, (m01 + m10) / s, (m02 + m20) / s, (m21 - m12) / s)
    if m11 > m22:
        s = math.sqrt(1.0 + m11 - m00 - m22) * 2
        return ((m01 + m10) / s, 0.25 * s, (m12 + m21) / s, (m02 - m20) / s)
    s = math.sqrt(1.0 + m22 - m00 - m11) * 2
    return ((m02 + m20) / s, (m12 + m21) / s, 0.25 * s, (m10 - m01) / s)


class FusionViz(Node):
    def __init__(self):
        super().__init__("fusion_viz")
        p = self.declare_parameter
        self.axes_spec = p("sound_axes", "forward=+z,left=-y,up=+x").value
        self.M = parse_axes(self.axes_spec)
        self.hfov = float(p("hfov_deg", 46.2).value)
        self.vfov = float(p("vfov_deg", 37.7).value)
        self.L = float(p("arrow_len", 2.5).value)
        self.F = float(p("fov_len", 3.0).value)
        rate = float(p("rate_hz", 10.0).value)
        self.pub = self.create_publisher(MarkerArray, "/fusion/viz", 10)
        self.create_subscription(String, "/drone_warning", self.on_warning, 10)
        self.create_subscription(String, "/fused/state", self.on_state, 10)
        self.sound = None          # (t, az, el, prob, xyz)
        self.trail = []            # (t, az, el)
        self.state = None          # (t, dict)
        self.create_timer(1.0 / rate, self.draw)
        # static TF camera -> odas: p_camera = M p_odas  -> rotation of the child (odas) in the parent (camera)
        self.tf = StaticTransformBroadcaster(self)
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id, t.child_frame_id = "camera", "odas"
        q = quat_from_rows(self.M)
        t.transform.rotation.x, t.transform.rotation.y, t.transform.rotation.z, t.transform.rotation.w = q
        self.tf.sendTransform(t)
        self.get_logger().info(f"/fusion/viz in frame 'camera' | sound_axes '{self.axes_spec}' | static TF camera->odas")

    # ---------------------------------------------------------------- inputs
    def on_warning(self, msg):
        xyz = warning_xyz(msg.data)
        if xyz is None:
            return
        az, el = xyz_to_camera(xyz, self.M)
        try:
            prob = float(json.loads(msg.data).get("prob") or 0.0)
        except (ValueError, AttributeError):
            prob = 0.0
        now = time.monotonic()
        self.sound = (now, az, el, prob, xyz)
        self.trail.append((now, az, el))
        self.trail = [x for x in self.trail if now - x[0] < 3.0]

    def on_state(self, msg):
        try:
            self.state = (time.monotonic(), json.loads(msg.data))
        except ValueError:
            pass

    # ---------------------------------------------------------------- drawing
    def mk(self, mid, mtype, color, scale=(0.05, 0.1, 0.1)):
        m = Marker()
        m.header.frame_id = "camera"
        m.header.stamp = self.get_clock().now().to_msg()
        m.ns, m.id, m.type, m.action = "fusion", mid, mtype, Marker.ADD
        m.pose.orientation.w = 1.0
        m.scale.x, m.scale.y, m.scale.z = (float(v) for v in scale)
        m.color = col(color)
        return m

    def delete(self, mid):
        m = Marker()
        m.header.frame_id = "camera"
        m.ns, m.id, m.action = "fusion", mid, Marker.DELETE
        return m

    def arrow(self, mid, az, el, length, color, width):
        m = self.mk(mid, Marker.ARROW, color, (width, width * 2.2, width * 3.0))
        u = unit(az, el)
        m.points = [Point(x=0.0, y=0.0, z=0.0), Point(x=u[0] * length, y=u[1] * length, z=u[2] * length)]
        return m

    def text(self, mid, x, y, z, s, color, size):
        m = self.mk(mid, Marker.TEXT_VIEW_FACING, color, (size, size, size))
        m.pose.position.x, m.pose.position.y, m.pose.position.z = x, y, z
        m.text = s
        return m

    def draw(self):
        now = time.monotonic()
        arr = MarkerArray()
        # ---- camera field of view (pyramid)
        fov = self.mk(0, Marker.LINE_LIST, GREY, (0.02, 0, 0))
        h, v = math.tan(math.radians(self.hfov / 2)) * self.F, math.tan(math.radians(self.vfov / 2)) * self.F
        c = [Point(x=self.F, y=s1 * h, z=s2 * v) for s1, s2 in ((1, 1), (-1, 1), (-1, -1), (1, -1))]
        o = Point(x=0.0, y=0.0, z=0.0)
        for i in range(4):
            fov.points += [o, c[i], c[i], c[(i + 1) % 4]]
        arr.markers.append(fov)
        arr.markers.append(self.text(1, self.F, 0.0, v + 0.25, f"CAMERA VIEW {self.hfov:.0f}x{self.vfov:.0f} deg", GREY, 0.18))
        # ---- sound
        fresh = self.sound is not None and now - self.sound[0] < 2.0
        if fresh:
            _, az, el, prob, xyz = self.sound
            arr.markers.append(self.arrow(2, az, el, self.L, YELLOW, 0.05))
            inside = abs(az) <= self.hfov / 2 and abs(el) <= self.vfov / 2
            u = unit(az, el)
            arr.markers.append(self.text(3, u[0] * (self.L + 0.4), u[1] * (self.L + 0.4), u[2] * (self.L + 0.4) + 0.2,
                                         f"SOUND p{prob:.2f}\n{words(az, el)}\n{'IN VIEW' if inside else 'OUT OF VIEW'}\n"
                                         f"x{xyz[0]:+.2f} y{xyz[1]:+.2f} z{xyz[2]:+.2f}", YELLOW, 0.16))
            # mirror point (z of the array flipped)
            maz, mel = xyz_to_camera((xyz[0], xyz[1], -xyz[2]), self.M)
            mu = unit(maz, mel)
            mir = self.mk(4, Marker.SPHERE, (0.6, 0.6, 0.6, 0.5), (0.18, 0.18, 0.18))
            mir.pose.position.x, mir.pose.position.y, mir.pose.position.z = (mu[0] * self.L, mu[1] * self.L, mu[2] * self.L)
            arr.markers.append(mir)
            tr = self.mk(5, Marker.SPHERE_LIST, (1.0, 0.85, 0.1, 0.5), (0.06, 0.06, 0.06))
            for _, a, e in self.trail:
                w = unit(a, e)
                tr.points.append(Point(x=w[0] * self.L, y=w[1] * self.L, z=w[2] * self.L))
            arr.markers.append(tr)
        else:
            arr.markers += [self.delete(2), self.delete(3), self.delete(4), self.delete(5)]
        # ---- fused output + state banner
        st = self.state[1] if self.state and now - self.state[0] < 2.0 else None
        if st and st.get("az_deg") is not None:
            colr = BLUE if st.get("source") == "vision" else GREEN
            arr.markers.append(self.arrow(6, st["az_deg"], st["el_deg"], self.L * 1.25, colr, 0.09))
        else:
            arr.markers.append(self.delete(6))
        if st:
            name = st.get("state", "?")
            src = {"vision": "VISION (YOLO + DyHiT)", "sound": "SOUND (UMA-8 + MobileNet)"}.get(st.get("source"), "none")
            colr = {"VISUAL_TRACK": BLUE, "SOUND_TRACK": GREEN}.get(name, WHITE)
            where = "" if st.get("az_deg") is None else f"\n{words(st['az_deg'], st['el_deg'])}"
            s = f"{name}\ndirection from {src}{where}\nvision {st.get('vision')}  sound {st.get('sound')}  target_lost {st.get('target_lost')}"
        else:
            colr, s = RED, "NO /fused/state\n(is the fusion running?)"
        arr.markers.append(self.text(7, 0.0, 0.0, 1.6, s, colr, 0.25))
        self.pub.publish(arr)


def main(args=None):
    rclpy.init(args=args)
    n = FusionViz()
    try:
        rclpy.spin(n)
    except KeyboardInterrupt:
        pass
    finally:
        n.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
