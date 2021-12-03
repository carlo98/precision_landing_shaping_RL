#!/usr/bin/env python3
import subprocess
import numpy as np
import rclpy
import time

# ROS dep
from px4_msgs.msg import VehicleOdometry
from std_msgs.msg import Int64


class GazeboRunnerNode:
    def __init__(self, node):
        self.node = node

        self.vehicle_odometry_subscriber = self.node.create_subscription(VehicleOdometry, '/fmu/vehicle_odometry/out',
                                                                         self.vehicle_odometry_callback, 1)
        self.resetting_publisher = self.node.create_publisher(Int64, '/env/resetting', 1)
        self.cont_takeoff_failing = 0
        self.started = False
        self.msg_reset_gazebo = Int64()
        self.start_gazebo()

    def vehicle_odometry_callback(self, obs):
        self.state = [obs.x, obs.y, obs.z, obs.vx, obs.vy, obs.vz]
        self.msg_reset_gazebo.data = 0  # Signaling to env that gazebo is ready
        self.resetting_publisher.publish(self.msg_reset_gazebo)
        if np.abs(self.state[2]) <= 0.6 and self.started:
            self.cont_takeoff_failing += 1
            if self.cont_takeoff_failing >= 4000:
                self.msg_reset_gazebo.data = 1  # Signaling to env that gazebo is resetting
                self.resetting_publisher.publish(self.msg_reset_gazebo)
                self.kill_gazebo()
                self.start_gazebo()
                self.cont_takeoff_failing = 0
                self.started = False
                self.msg_reset_gazebo.data = 0  # Signaling to env that gazebo is ready
                self.resetting_publisher.publish(self.msg_reset_gazebo)
        else:
            self.started = True
            self.cont_takeoff_failing = 0

    def start_gazebo(self):
        self.gazebo = subprocess.Popen(["make", "px4_sitl_rtps", "gazebo"], cwd="/src/shared/PX4-Autopilot")
        time.sleep(5)

    def kill_gazebo(self):
        self.gazebo.kill()
        time.sleep(1)


if __name__ == '__main__':
    rclpy.init(args=None)
    m_node = rclpy.create_node('gazebo_runner_node')
    gsNode = GazeboRunnerNode(m_node)
    rclpy.spin(m_node)
