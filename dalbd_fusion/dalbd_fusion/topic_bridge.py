#!/usr/bin/env python3
"""topic_bridge.py - JETSON A: forward only LIGHT topics from the DALBD domains (vision 0, sound 42) to the domain
of another machine (Jetson B, default 5). The raw camera /image (~1 MB per frame) never leaves Jetson A, so a
slow reader on another machine can no longer slow the pipeline down.

    domain 0  /image/compressed /target/bearing /target/status /fused/state /fused/bearing ┐
    domain 42 /drone_warning /target/direction /raw/sst /fused/state                         ├──> domain 5 (same names)
                                                                                             ┘
Messages are forwarded as raw bytes (no decoding), with the type found at run time. Towards the other machine the
QoS is BEST_EFFORT: if it is slow, only ITS copies are dropped; Jetson A is never held back.

Run (JETSON container, any domain):  ros2 run dalbd_fusion topic_bridge
Options (name:=value):
  to_domain:=5
  topics:="0:/image/compressed,0:/target/bearing,0:/target/status,0:/fused/state,0:/fused/bearing,42:/drone_warning,42:/target/direction,42:/raw/sst"
  rename:="42:/fused/state=/fused/state_sound"     (optional: give a forwarded topic another name in the target domain)
Every 5 s it prints the rate of each forwarded topic.
Jetson B:  export ROS_DOMAIN_ID=5; ros2 topic list
           raw image if needed (decoded ON Jetson B):
           ros2 run image_transport republish compressed raw --ros-args -r in/compressed:=/image/compressed -r out:=/image
"""
import sys
import threading
import time

DEFAULT_TOPICS = ("0:/image/compressed,0:/target/bearing,0:/target/status,0:/fused/state,0:/fused/bearing,"
                  "42:/drone_warning,42:/target/direction,42:/raw/sst")


def _arg(name, default):
    for a in sys.argv:
        if a.startswith(name + ":="):
            return a.split(":=", 1)[1].strip("\"'")
    return default


def parse_topics(text):
    """'0:/a,42:/b' -> [(0, '/a'), (42, '/b')]"""
    out = []
    for part in text.replace(" ", "").split(","):
        if not part:
            continue
        d, t = part.split(":", 1)
        out.append((int(d), t if t.startswith("/") else "/" + t))
    return out


def parse_rename(text):
    """'42:/fused/state=/fused/state_sound' -> {(42, '/fused/state'): '/fused/state_sound'}"""
    out = {}
    for part in text.replace(" ", "").split(","):
        if "=" in part:
            src, dst = part.split("=", 1)
            d, t = src.split(":", 1)
            out[(int(d), t)] = dst
    return out


def main(args=None):
    import rclpy
    from rclpy.context import Context
    from rclpy.executors import SingleThreadedExecutor
    from rclpy.node import Node
    from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message

    to_dom = int(_arg("to_domain", "5"))
    pairs = parse_topics(_arg("topics", DEFAULT_TOPICS))
    rename = parse_rename(_arg("rename", ""))
    if to_dom in {d for d, _ in pairs}:
        sys.exit(f"[FAIL] to_domain {to_dom} is also a source domain: choose another one (e.g. 5)")

    sub_qos = QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=5, reliability=ReliabilityPolicy.BEST_EFFORT,
                         durability=DurabilityPolicy.VOLATILE)    # BEST_EFFORT reader matches any publisher
    pub_qos = QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=5, reliability=ReliabilityPolicy.BEST_EFFORT,
                         durability=DurabilityPolicy.VOLATILE)    # never held back by the other machine

    ctx = {}
    for d in sorted({d for d, _ in pairs} | {to_dom}):
        c = Context()
        rclpy.init(args=sys.argv, context=c, domain_id=d)
        ctx[d] = c
    out_node = Node("dalbd_topic_bridge_out", context=ctx[to_dom])
    counts, lock = {}, threading.Lock()

    class Src(Node):
        def __init__(self, dom, topics):
            super().__init__(f"dalbd_topic_bridge_in_{dom}", context=ctx[dom])
            self.dom, self.topics, self.links = dom, topics, {}
            self.create_timer(1.0, self.discover)

        def discover(self):
            types = dict(self.get_topic_names_and_types())
            for t in self.topics:
                if t in self.links or t not in types:
                    continue
                tname = types[t][0]
                try:
                    mtype = get_message(tname)
                except (AttributeError, ModuleNotFoundError, ValueError) as e:
                    self.get_logger().warn(f"{t}: cannot load {tname} ({e}) - source its workspace first")
                    continue
                dst = rename.get((self.dom, t), t)
                pub = out_node.create_publisher(mtype, dst, pub_qos)
                key = f"{self.dom}:{t}"
                counts[key] = 0

                def cb(raw, pub=pub, key=key, mtype=mtype):
                    try:
                        pub.publish(raw)           # serialized bytes in, serialized bytes out (no decoding)
                    except TypeError:              # older rclpy without raw publish: decode once
                        pub.publish(deserialize_message(raw, mtype))
                    with lock:
                        counts[key] += 1
                self.links[t] = self.create_subscription(mtype, t, cb, sub_qos, raw=True)
                print(f"  bridge domain {self.dom} {t} [{tname}] -> domain {to_dom} {dst}", flush=True)

    by_dom = {}
    for d, t in pairs:
        by_dom.setdefault(d, []).append(t)
    srcs = [Src(d, ts) for d, ts in by_dom.items()]

    def report():
        with lock:
            snap = dict(counts)
            for k in counts:
                counts[k] = 0
        if snap:
            print(time.strftime("%H:%M:%S") + " | " + " | ".join(f"{k} {v / 5.0:.1f}Hz" for k, v in sorted(snap.items())),
                  flush=True)
    out_node.create_timer(5.0, report)

    exs, threads = [], []
    for n in srcs + [out_node]:
        ex = SingleThreadedExecutor(context=n.context)
        ex.add_node(n)
        exs.append(ex)
    for ex in exs[:-1]:
        th = threading.Thread(target=ex.spin, daemon=True)
        th.start()
        threads.append(th)
    print(f"topic_bridge: {', '.join(f'{d}:{t}' for d, t in pairs)} -> domain {to_dom} (BEST_EFFORT)\n"
          f"  waiting for the topics to appear...", flush=True)
    try:
        exs[-1].spin()
    except KeyboardInterrupt:
        pass
    finally:
        for ex in exs:
            ex.shutdown()
        for n in srcs + [out_node]:
            n.destroy_node()
        for c in ctx.values():
            if c.ok():
                rclpy.shutdown(context=c)


if __name__ == "__main__":
    main()
