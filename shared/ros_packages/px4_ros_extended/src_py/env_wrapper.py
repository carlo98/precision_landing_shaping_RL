#!/usr/bin/env python3
import numpy as np
from rewards import Reward

# ROS dep
from std_msgs.msg import Int64
from std_msgs.msg import Float32MultiArray
import rclpy

# Gazebo
from gazebo_msgs.msg import ContactsState
from gazebo_msgs.msg import ModelStates


class EnvWrapperNode:
    def __init__(self, node, state_shape, action_shape, max_height, max_side, max_vel_z, max_vel_xy, eps_pos_xy):
        self.node = node

        imu_qos = rclpy.qos.QoSPresetProfiles.get_from_short_key('sensor_data')  # Quality of Service
        self.bumper_subscriber = self.node.create_subscription(ContactsState, "/bumper_iris",
                                                               self.check_landed_callback, imu_qos)
        self.vehicle_odometry_subscriber = self.node.create_subscription(Float32MultiArray, '/agent/odom',
                                                                         self.vehicle_odometry_callback, 1)
        self.agent_vel_publisher = self.node.create_publisher(Float32MultiArray, "/agent/velocity", 1)
        self.play_reset_publisher = self.node.create_publisher(Float32MultiArray, "/env/play_reset/in", 1)
        self.play_reset_subscriber = self.node.create_subscription(Float32MultiArray, "/env/play_reset/out",
                                                                   self.play_reset_callback, 1)
        self.agent_action_received_subscriber = self.node.create_subscription(Int64, "/agent/action_received",
                                                                              self.action_received_callback, 1)
        self.gazebo_down_publisher = self.node.create_publisher(Int64, "/env/gazebo_down", 1)
        self.state_world_subscriber = self.node.create_subscription(ModelStates, '/gazebo/model_states_gazebo',
                                                                    self.model_callback, imu_qos)

        self.reward = Reward(max_height, max_side)

        self.action_received = False
        self.state = np.zeros(state_shape)
        self.state_world = np.zeros(state_shape)
        self.ir_beacon_state = np.zeros(state_shape)
        self.landed = False
        self.landed_received = False
        self.reset = True
        self.play = False

        self.eps_pos_xy = eps_pos_xy
        self.eps_vel_xy = 0.05
        self.max_vel_z = max_vel_z
        self.max_vel_xy = max_vel_xy
        self.state_shape = state_shape
        self.action_shape = action_shape

    def vehicle_odometry_callback(self, obs):
        self.state = -np.array(obs.data[:self.state_shape])
    
    def model_callback(self, msg):
        drone_id = 0
        if "iris_irlock" in msg.name:
            while msg.name != "iris_irlock":
                drone_id += 1
            self.state_world[2] = -1.0*msg.pose[drone_id].position.z
            self.state_world[0] = msg.pose[drone_id].position.y
            self.state_world[1] = msg.pose[drone_id].position.x
            self.state_world[5] = -1.0*msg.twist[drone_id].linear.z
            self.state_world[3] = msg.twist[drone_id].linear.y
            self.state_world[4] = msg.twist[drone_id].linear.x
            print("Iris: ", self.state_world)
        
        beacon_id = 0
        if "irlock_beacon" in msg.name:
            while msg.name != "irlock_beacon":
                beacon_id += 1
            self.ir_beacon_state[0] = msg.pose[beacon_id].position.y
            self.ir_beacon_state[1] = msg.pose[beacon_id].position.x
            self.ir_beacon_state[2] = -1.0*msg.pose[beacon_id].position.z
            # self.ir_beacon_orientation = msg.pose[beacon_id].orientation
            self.ir_beacon_state[3] = msg.twist[beacon_id].linear.y
            self.ir_beacon_state[4] = msg.twist[beacon_id].linear.x
            self.ir_beacon_state[5] = -1.0*msg.twist[beacon_id].linear.z
            # self.ir_beacon_twist_angular = msg.twist[beacon_id].angular
            print("Beacon: ", self.ir_beacon_state)

    def act(self, action, normalize):
    
        self.landed_received = False

        if not self.landed:
            if self.action_shape == 2:  # Predicting vx and vy
                abs_height = np.abs(self.state_world[2])
                vel_z = 0.6 if abs_height > 1.0 else 0.9*abs_height
                if abs_height <= 0.50:
                    vel_z = 0.25
            elif self.action_shape == 3:  # Predicting vx, vy and vz
                vel_z = action[2] * self.max_vel_z
            action_msg = Float32MultiArray()
            action_msg.data = [-action[0] * self.max_vel_xy, -action[1] * self.max_vel_xy, -vel_z]  # Fixed z velocity
            self.agent_vel_publisher.publish(action_msg)

            while not self.action_received:  # Wait for confirmation from environment
                pass

            self.state_world = np.zeros(self.state_shape)
            while (self.state_world == 0).all():  # Wait for first message to arrive
                pass
        
        new_state = np.copy(self.state_world - self.ir_beacon_state)
        reward, done = self.reward.get_reward(new_state, normalize(np.copy(new_state)), action, self.landed, self.eps_pos_xy, self.eps_vel_xy)
        print(new_state, " ", reward)

        self.action_received = False
        return new_state, reward, done

    def action_received_callback(self, msg):
        self.action_received = msg.data == 1

    def check_landed_callback(self, msg):
        self.landed = len(msg.states) > 0
        self.landed_received = True

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
        # self.reward.init_shaping(self.state_world)  # Initialising shaping for reward
        self.play_reset_publisher.publish(play_reset_msg)
        self.play = True

        self.state_world = np.zeros(self.state_shape)
        while (self.state_world == 0).all():  # Wait for first message to arrive
            pass

        return np.copy(self.state_world)
        
    def play_reset_callback(self, msg):  # Used for synchronization with gazebo
        if msg.data[1] == 0:
            self.reset = False

    def shutdown_gazebo(self):
        """
        Shutdown message to Gazebo Node
        """
        msg = Int64()
        msg.data = 1
        self.gazebo_down_publisher.publish(msg)
