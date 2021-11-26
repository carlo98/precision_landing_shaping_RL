#!/usr/bin/env python3
import numpy as np

# ROS dep
import rclpy
import rospy

# ROS
from std_msgs.msg import Int64

# PX4 msgs
from px4_msgs.msg import VehicleOdometry
from px4_msgs.msg import Timesync

# Custom msgs
from custom_msgs.msg import Float32MultiArray


class EnvWrapperNode:
    def __init__(self, node):
        self.node = node

        self.vehicle_odometry_subscriber = self.node.create_subscription(VehicleOdometry, 'fmu/vehicle_odometry/out', self.vehicle_odometry_callback, 1)
        self.agent_vel_publisher = self.node.create_publisher(Float32MultiArray, "/agent/velocity", 1)
        self.agent_action_received_publisher = self.node.create_subscription(Int64, "/agent/action_received", self.action_received_callback, 1)
        self.timesync_sub_ = self.node.create_subscription(Timesync, "fmu/timesync/out", self.timestamp_callback, 1)

        self.action_received = False
        self.timestamp_ = 0.0
        self.state = self.previous_state = np.zeros(9)

    def vehicle_odometry_callback(self, obs):
        self.state = np.array(obs.px, obs.py, obs.pz, obs.vx, obs.vy, obs.vz, obs.wx, obs.wy, obs.wz)

    def timestamp_callback(self, msg):
        self.timestamp_ = msg.timestamp

    def act(self, action):
        action_msg = Float32MultiArray()
        action_msg[0] = action[0]
        action_msg[1] = action[1]
        action_msg[2] = action[2]
        self.vehicle_odometry_subscriber.publish(action_msg)

        while not self.action_received:  # Wait for confirmation from environment
            pass

        reward, done = self.compute_reward(self.state, self.previous_state)

        self.previous_state = self.state
        self.action_received = False
        return self.state, reward, done

    def action_received_callback(self, msg):
        self.action_received = msg

    def compute_reward(self, obs, previous_obs):
        return np.sum(obs-previous_obs)


if __name__ == '__main__':
    rclpy.init(args=None)
    m_node = rclpy.create_node('gs_node')
    gsNode = EnvWrapperNode(m_node)
    rclpy.spin(m_node)
