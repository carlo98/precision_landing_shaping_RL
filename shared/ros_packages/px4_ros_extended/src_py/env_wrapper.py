#!/usr/bin/env python3
import numpy as np

# ROS dep
import rclpy

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

        self.vehicle_odometry_subscriber = self.node.create_subscription(VehicleOdometry, 'fmu/vehicle_odometry/out',
                                                                         self.vehicle_odometry_callback, 1)
        self.agent_vel_publisher = self.node.create_publisher(Float32MultiArray, "/agent/velocity", 1)
        self.play_reset_publisher = self.node.create_publisher(Float32MultiArray, "/env/play_reset", 1)
        self.play_reset_subscriber = self.node.create_subscription(Float32MultiArray, "/env/play_reset", self.play_reset_callback, 1)
        self.agent_action_received_publisher = self.node.create_subscription(Int64, "/agent/action_received",
                                                                             self.action_received_callback, 1)
        self.timesync_sub_ = self.node.create_subscription(Timesync, "fmu/timesync/out", self.timestamp_callback, 1)

        self.action_received = False
        self.timestamp_ = 0.0
        self.state = self.previous_state = np.zeros(9)
        self.collision = False
        self.current_shaping = 0.0
        self.reset = True

        self.eps_pos_z = 0.05
        self.eps_pos_xy = 0.1
        self.eps_vel_z = 0.1
        self.eps_vel_xy = 0.1

        self.min_reward = -1
        # Weights for pos, velocity, angular velocity, action, 3 x single action
        self.coeffs = np.array([-100, -10, -10, -1, 10, 10, 10])
        self.stop_reward = 10

    def vehicle_odometry_callback(self, obs):
        self.state = np.array([-obs.x, -obs.y, -obs.z, -obs.vx, -obs.vy, -obs.vz, obs.rollspeed, obs.pitchspeed, obs.yawspeed])
        self.collision = self.check_collision()

    def timestamp_callback(self, msg):
        self.timestamp_ = msg.timestamp

    def act(self, action):
        if self.collision:
            return self.state, self.min_reward, True

        action_msg = Float32MultiArray()
        action_msg.data = [action[0], action[1], action[2]]
        self.agent_vel_publisher.publish(action_msg)
        
        print("Waiting for action received")

        while not self.action_received:  # Wait for confirmation from environment
            pass

        reward, done = self.compute_reward(self.state, action)
        
        print("Reward: ", reward)

        self.previous_state = self.state
        self.action_received = False
        return self.state, reward, done

    def action_received_callback(self, msg):
        self.action_received = msg

    def compute_reward(self, obs, action):
        collision = self.check_collision()
        if collision:
            return self.min_reward, True

        done = False
        if obs[2] <= self.eps_pos_z and obs[5] <= self.eps_vel_z \
                and obs[3] <= self.eps_vel_xy and obs[4] <= self.eps_vel_xy:
            done = True

        shaping = self.coeffs[0] * np.sqrt(obs[0] ** 2 + obs[1] ** 2 + obs[2] ** 2) + \
                  self.coeffs[1] * np.sqrt(obs[3] ** 2 + obs[4] ** 2 + obs[5] ** 2) + \
                  self.coeffs[2] * np.sqrt(action[0] ** 2 + action[1] ** 2 + action[2] ** 2) + \
                  self.coeffs[3] * self.stop_reward * (1 - np.abs(action[0])) + \
                  self.coeffs[4] * self.stop_reward * (1 - np.abs(action[1])) + \
                  self.coeffs[5] * self.stop_reward * (1 - np.abs(action[2]))

        reward = shaping - self.current_shaping
        self.current_shaping = shaping

        return reward, done

    def check_collision(self):
        return self.state[2] <= self.eps_pos_z and \
               (self.state[5] > self.eps_vel_z or self.state[3] > self.eps_vel_xy or self.state[4] > self.eps_vel_xy)

    def reset_env(self):
        play_reset_msg = Float32MultiArray()
        play_reset_msg.data = [0.0, 1.0]
        self.play_reset_publisher.publish(play_reset_msg)
        self.reset = True

        while self.reset:  # Wait for takeoff completion
            pass

    def play_reset_callback(self, msg):
        if msg.data[1] == 0 and msg.data[0] == 1:
            self.reset = False


if __name__ == '__main__':
    rclpy.init(args=None)
    m_node = rclpy.create_node('gs_node')
    gsNode = EnvWrapperNode(m_node)
    rclpy.spin(m_node)
