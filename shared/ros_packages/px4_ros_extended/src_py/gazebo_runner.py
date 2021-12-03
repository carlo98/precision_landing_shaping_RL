#!/usr/bin/env python3
import subprocess
import numpy as np
import rclpy
import time

# ROS dep
from px4_msgs.msg import VehicleOdometry


class GazeboRunnerNode:
    def __init__(self, node):
        self.node = node

        self.vehicle_odometry_subscriber = self.node.create_subscription(VehicleOdometry, '/fmu/vehicle_odometry/out',
                                                                         self.vehicle_odometry_callback, 1)
        self.cont_takeoff_failing = 0
        self.started = False
        self.start_gazebo()

    def vehicle_odometry_callback(self, obs):
        self.state = [obs.x, obs.y, obs.z, obs.vx, obs.vy, obs.vz]
        if np.abs(self.state[2]) <= 0.6 and self.started:
            self.cont_takeoff_failing += 1
            if self.cont_takeoff_failing >= 20000:
                self.kill_gazebo()
                self.start_gazebo()
                self.cont_takeoff_failing = 0
                self.started = False
        else:
            self.started = True
            self.cont_takeoff_failing = 0

    def start_gazebo(self):
        self.gazebo = subprocess.Popen(["make", "px4_sitl_rtps", "gazebo"], cwd="/src/shared/PX4-Autopilot")

    def kill_gazebo(self):
        self.gazebo.kill()
        time.sleep(2)


if __name__ == '__main__':
    rclpy.init(args=None)
    m_node = rclpy.create_node('gazebo_runner_node')
    gsNode = GazeboRunnerNode(m_node)
    rclpy.spin(m_node)
