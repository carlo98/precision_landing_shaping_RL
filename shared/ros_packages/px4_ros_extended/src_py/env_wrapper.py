#!/usr/bin/env python3
import numpy as np
from rewards import Reward
import rclpy

# ROS dep
from std_msgs.msg import Int64
from gazebo_msgs.msg import Empty
from gazebo_msgs.msg import EntityState
from gazebo_msgs.srv import SpawnEntity
from gazebo_msgs.srv import DeleteEntity
from geometry_msgs.msg import Pose
from geometry_msgs.msg import Twist

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
        self.timestamp_ = 0.0
        self.state = np.zeros(state_shape)
        self.collision = False
        self.reset = True
        self.play = False

        self.eps_pos_z = 0.08
        self.eps_pos_xy = 0.50  # Drone can land on a 1x1 (m) target
        self.eps_vel_xy = 0.05
        self.max_vel_z = max_vel_z
        self.max_vel_xy = max_vel_xy
        self.state_shape = state_shape
        self.respawn_flag = False

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
        self.reset_world()
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

    # The following methods are not used, because they create problems with PX4 calibration
    def set_pose_model(self):
        self.service_call("/pause_physics")
        self.service_call_set_state("/demo/set_entity_state")
        self.service_call("/unpause_physics")

    def reset_world(self):
        self.service_call("/pause_physics")
        self.service_call("/reset_world")
        self.service_call("/unpause_physics")

    def respawn(self):
        self.respawn_flag = True
        #self.service_call("/pause_physics")
        self.service_delete("/delete_entity")
        self.service_spawn("/spawn_entity")
        #self.service_call("/unpause_physics")
        self.respawn_flag = False

    def service_call(self, name_service):
        respawn = self.node.create_client(Empty, name_service)
        while not respawn.wait_for_service(timeout_sec=1.0):
            self.node.get_logger().info('service not available, waiting again...')
        req = Empty.Request()
        fut = respawn.call_async(req)
        while not fut.done():
            pass

    def service_call_set_state(self, name_service):
        state_msg = EntityState()
        pose_msg = Pose()
        twist_msg = Twist()
        pose_msg.position.x = 0.0
        pose_msg.position.y = 0.0
        pose_msg.position.z = -0.21
        pose_msg.orientation.x = 0.0
        pose_msg.orientation.y = 0.0
        pose_msg.orientation.z = 0.0
        pose_msg.orientation.w = 0.0
        twist_msg.linear.x = 0.0
        twist_msg.linear.y = 0.0
        twist_msg.linear.z = 0.0
        twist_msg.angular.x = 0.0
        twist_msg.angular.y = 0.0
        twist_msg.angular.z = 0.0
        state_msg.name = 'iris'
        state_msg.pose = pose_msg
        state_msg.twist = twist_msg
        respawn = self.node.create_client(SetEntityState, name_service)
        while not respawn.wait_for_service(timeout_sec=1.0):
            self.node.get_logger().info('service not available, waiting again...')
        req = SetEntityState.Request(state=state_msg)
        fut = respawn.call_async(req)
        while not fut.done():
            pass

    def service_spawn(self, name_service):
        f_sdf = open("/src/shared/PX4-Autopilot/Tools/sitl_gazebo/models/iris/iris.sdf", "r")
        sdf = f_sdf.read()
        f_sdf.close()
        print(sdf)
        pose_msg = Pose()
        pose_msg.position.x = 0.0
        pose_msg.position.y = 0.0
        pose_msg.position.z = 0.2
        pose_msg.orientation.x = 0.0
        pose_msg.orientation.y = 0.0
        pose_msg.orientation.z = 0.0
        pose_msg.orientation.w = 0.0
        respawn = self.node.create_client(SpawnEntity, name_service)
        while not respawn.wait_for_service(timeout_sec=1.0):
            self.node.get_logger().info(name_service + ': service not available, waiting again...')
        req = SpawnEntity.Request(name='iris', xml=sdf, robot_namespace='', reference_frame="world",
                                  initial_pose=pose_msg)
        fut = respawn.call_async(req)
        rclpy.spin_until_future_complete(self.node, fut)
        while not fut.done():
            print(name_service)

    def service_delete(self, name_service):
        respawn = self.node.create_client(DeleteEntity, name_service)
        while not respawn.wait_for_service(timeout_sec=1.0):
            self.node.get_logger().info(name_service + ': service not available, waiting again...')
        req = DeleteEntity.Request(name="iris")
        fut = respawn.call_async(req)
        rclpy.spin_until_future_complete(self.node, fut)
        #while not fut.done():
        #    print(name_service)


