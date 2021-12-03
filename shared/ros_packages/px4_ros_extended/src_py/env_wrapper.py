#!/usr/bin/env python3
import numpy as np
from rewards import Reward
import rclpy

# ROS dep
from std_msgs.msg import Int64

# Custom msgs
from custom_msgs.msg import Float32MultiArray


class EnvWrapperNode:
    def __init__(self, node, state_shape, max_height, max_side, max_vel_z, max_vel_xy):
        self.node = node

        self.vehicle_odometry_subscriber = self.node.create_subscription(Float32MultiArray, 'agent/odom',
                                                                         self.vehicle_odometry_callback, 1)
        self.agent_vel_publisher = self.node.create_publisher(Float32MultiArray, "/agent/velocity", 1)
        self.play_reset_publisher = self.node.create_publisher(Float32MultiArray, "/env/play_reset/in", 1)
        self.play_reset_subscriber = self.node.create_subscription(Float32MultiArray, "/env/play_reset/out",
                                                                   self.play_reset_callback, 1)
        self.agent_action_received_publisher = self.node.create_subscription(Int64, "/agent/action_received",
                                                                             self.action_received_callback, 1)

        self.reward = Reward(max_height, max_side)

        self.action_received = False
        self.state = np.zeros(state_shape)
        self.collision = False
        self.reset = True
        self.play = False

        self.eps_pos_z = 0.08
        self.eps_pos_xy = 0.80  # Drone can land on a 1x1 (m) target
        self.eps_vel_xy = 0.05
        self.max_vel_z = max_vel_z
        self.max_vel_xy = max_vel_xy
        self.state_shape = state_shape

    def vehicle_odometry_callback(self, obs):
        # 0--x, 1--y, 2--z, 3--vx, 4--vy
        self.state = -np.array(obs.data)
        self.collision = self.check_collision()

    def act(self, action, normalize):

        if not self.collision:
            vel_z = 0.6 if self.state[2] > 1.0 else 0.9*self.state[2]
            if np.abs(self.state[2]) <= 0.25:
                vel_z = np.float64(self.state[2])
            action_msg = Float32MultiArray()
            action_msg.data = [-action[0] * self.max_vel_xy, -action[1] * self.max_vel_xy, -vel_z]  # Fixed z velocity
            self.agent_vel_publisher.publish(action_msg)

            while not self.action_received:  # Wait for confirmation from environment
                pass

            self.state = np.zeros(self.state_shape)
            while (self.state == 0).all():  # Wait for first message to arrive
                pass

        new_state = np.copy(self.state)
        reward, done = self.reward.get_reward(new_state, normalize(np.copy(new_state)), action, self.eps_pos_z, self.eps_pos_xy, self.eps_vel_xy)

        self.action_received = False
        return new_state, reward, done

    def action_received_callback(self, msg):
        self.action_received = msg.data == 1

    def check_collision(self):
        return self.play and np.abs(self.state[2]) <= self.eps_pos_z and \
               (np.abs(self.state[3]) > self.eps_vel_xy or np.abs(self.state[4]) > self.eps_vel_xy)

    def reset_env(self):  # Used for synchronization with gazebo
        play_reset_msg = Float32MultiArray()
        play_reset_msg.data = [1.0, 1.0]
        self.play_reset_publisher.publish(play_reset_msg)

        self.reset = True
        while self.reset:  # Wait for takeoff completion
            pass
        self.play = False
            
    def play_env(self):  # Used for synchronization with gazebo
        play_reset_msg = Float32MultiArray()
        play_reset_msg.data = [1.0, 0.0]
        # self.reward.init_shaping(self.state)  # Initialising shaping for reward
        self.play_reset_publisher.publish(play_reset_msg)
        self.play = True

        self.state = np.zeros(self.state_shape)
        while (self.state == 0).all():  # Wait for first message to arrive
            pass

        return np.copy(self.state)
        
    def play_reset_callback(self, msg):  # Used for synchronization with gazebo
        if msg.data[1] == 0:
            self.reset = False
