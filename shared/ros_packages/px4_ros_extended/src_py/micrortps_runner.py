#!/usr/bin/env python3
import subprocess
import numpy as np
import time
import rclpy

# ROS dep
from px4_msgs.msg import VehicleOdometry


class MicrortpsRunnerNode:
    def __init__(self, node):
        self.node = node

        self.vehicle_odometry_subscriber = self.node.create_subscription(VehicleOdometry, '/fmu/vehicle_odometry/out',
                                                                         self.vehicle_odometry_callback, 1)
        self.cont_takeoff_failing = 0
        self.started = False
        self.start_micrortps()

    def vehicle_odometry_callback(self, obs):
        self.state = [obs.x, obs.y, obs.z, obs.vx, obs.vy, obs.vz]
        if np.abs(self.state[2]) <= 0 and self.started:
            self.cont_takeoff_failing += 1
            if self.cont_takeoff_failing >= 20000:
                self.kill_micrortps()
                self.start_micrortps()
                self.cont_takeoff_failing = 0
                self.started = False
        else:
            self.started = True
            self.cont_takeoff_failing = 0

    def start_micrortps(self):
        self.micrortps = subprocess.Popen(["micrortps_agent", "-t", "UDP"])

    def kill_micrortps(self):
        self.micrortps.kill()
        time.sleep(2)


if __name__ == '__main__':
    rclpy.init(args=None)
    m_node = rclpy.create_node('micrortps_runner_node')
    gsNode = MicrortpsRunnerNode(m_node)
    rclpy.spin(m_node)
