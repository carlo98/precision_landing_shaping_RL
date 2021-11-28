#!/usr/bin/env python3
import numpy as np
from rewards import Reward

# ROS dep
import rclpy
from std_msgs.msg import Int64

# PX4 msgs
from px4_msgs.msg import VehicleOdometry
from px4_msgs.msg import Timesync

# Custom msgs
from custom_msgs.msg import Float32MultiArray


class EnvWrapperNode:
    def __init__(self, node):
        self.node = node

        self.vehicle_odometry_subscriber = self.node.create_subscription(VehicleOdometry, 'fmu/vehicle_odometry/out',
                                                                         self.vehicle_odometry_callback, 1)
        self.agent_vel_publisher = self.node.create_publisher(Float32MultiArray, "/agent/velocity", 1)
        self.play_reset_publisher = self.node.create_publisher(Float32MultiArray, "/env/play_reset/in", 1)
        self.play_reset_subscriber = self.node.create_subscription(Float32MultiArray, "/env/play_reset/out", self.play_reset_callback, 1)
        self.agent_action_received_publisher = self.node.create_subscription(Int64, "/agent/action_received",
                                                                             self.action_received_callback, 1)
        self.timesync_sub_ = self.node.create_subscription(Timesync, "fmu/timesync/out", self.timestamp_callback, 1)

        self.reward = Reward()

        self.action_received = False
        self.timestamp_ = 0.0
        self.state = self.previous_state = np.zeros(9)
        self.collision = False
        self.reset = True
        self.play = False

        self.eps_pos_z = 0.15
        self.eps_pos_xy = 0.15
        self.eps_vel_z = 0.1
        self.eps_vel_xy = 0.1
        self.max_vel_z = 1.5
        self.max_vel_xy = 1.5

    def vehicle_odometry_callback(self, obs):
        self.state = np.array([-obs.x, -obs.y, -obs.z, -obs.vx, -obs.vy, -obs.vz])  # obs.rollspeed, obs.pitchspeed, obs.yawspeed])
        self.collision = self.check_collision()

    def timestamp_callback(self, msg):
        self.timestamp_ = msg.timestamp

    def act(self, action):

        if not self.collision:
            action_msg = Float32MultiArray()
            action_msg.data = [action[0] * self.max_vel_xy, action[1] * self.max_vel_xy, action[2] * self.max_vel_z]
            self.agent_vel_publisher.publish(action_msg)

            while not self.action_received:  # Wait for confirmation from environment
                pass

        reward, done = self.reward.get_reward(self.state, action, self.eps_pos_z, self.eps_pos_xy, self.eps_vel_z, self.eps_vel_xy)

        self.previous_state = self.state
        self.action_received = False
        return self.state, reward, done

    def action_received_callback(self, msg):
        self.action_received = msg

    def check_collision(self):
        return self.play and np.abs(self.state[2]) <= self.eps_pos_z and \
               (np.abs(self.state[5]) > self.eps_vel_z or np.abs(self.state[3]) > self.eps_vel_xy or np.abs(self.state[4]) > self.eps_vel_xy)

    def reset_env(self):  # Used for synchronization with gazebo
        play_reset_msg = Float32MultiArray()
        play_reset_msg.data = [0.0, 1.0]
        self.play_reset_publisher.publish(play_reset_msg)
        self.reset = True
        self.play = False

        while self.reset:  # Wait for takeoff completion
            pass
            
    def play_env(self):  # Used for synchronization with gazebo
        play_reset_msg = Float32MultiArray()
        play_reset_msg.data = [1.0, 0.0]
        self.previous_state = self.state  # Reset previous state
        self.reward.init_shaping(self.state)  # Initialising shaping for reward
        self.play_reset_publisher.publish(play_reset_msg)
        self.play = True
        
    def play_reset_callback(self, msg):  # Used for synchronization with gazebo
        if msg.data[1] == 0:
            self.reset = False


if __name__ == '__main__':
    rclpy.init(args=None)
    m_node = rclpy.create_node('gs_node')
    gsNode = EnvWrapperNode(m_node)
    rclpy.spin(m_node)
