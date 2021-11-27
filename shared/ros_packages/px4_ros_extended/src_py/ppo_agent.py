#!/usr/bin/env python3
# Python dep
import yaml
import numpy as np
from collections import deque
import time
import torch

# ROS dep
import rclpy

# PX4 msgs
from px4_msgs.msg import VehicleOdometry
from px4_msgs.msg import Timesync

# Custom msgs
from custom_msgs.msg import Float32MultiArray

# Agent and PPO
from PPO.model import Policy
from PPO.ppo import PPO
from PPO.memory import Memory


class AgentNode:
    def __init__(self, node):
        self.node = node

        with open('params.yaml') as info:
            self.info_dict = yaml.load(info)

        self.vehicle_odometry_subscriber = self.node.create_subscription(VehicleOdometry, 'fmu/vehicle_command/in', self.vehicle_odometry_callback, 1)
        self.vehicle_odometry_subscriber = self.node.create_subscription(Float32MultiArray, 'fmu/vehicle_command/in', self.vehicle_odometry_callback, 1)
        self.agent_vel_publisher = self.node.create_publisher(Float32MultiArray, "/agent/velocity", 1)

        self.timesync_sub_ = self.node.create_subscription(Timesync, "fmu/timesync/out", self.timestamp_callback, 1)

        self.timestamp_ = 0.0

        torch.manual_seed(self.info_dict['seed'])
        torch.cuda.manual_seed_all(self.info_dict['seed'])

        self.agents = Policy(obs_shape=self.info_dict['obs_shape'], action_space=self.info_dict['action_space'])
        self.ppo = PPO(self.agents, self.info_dict['clip_param'], self.info_dict['ppo_epoch'],
                       self.info_dict['num_mini_batch'], self.info_dict['value_loss_coef'],
                       self.info_dict['entropy_coef'], lr=self.info_dict['lr'], eps=self.info_dict['eps'],
                       max_grad_norm=self.info_dict['max_grad_norm'])

        self.memory = Memory(self.info_dict['num-steps'], self.info_dict['num-processes'],
                             self.info_dict['obs_shape'], self.info_dict['action_space'])

        self.episode_rewards = deque(maxlen=10)
        self.start = time.time()

        self.cont_steps = 0
        self.previous_obs = np.zeros(self.info_dict['obs_shape'])
        self.previous_action = np.zeros(self.info_dict['action_space'])
        self.previous_action_log_prob = np.zeros(self.info_dict['action_space'])
        self.previous_value = 0.0

    def vehicle_odometry_callback(self, obs):
        inputs = np.array(obs.px, obs.py, obs.pz, obs.vx, obs.vy, obs.vz, obs.wx, obs.wy, obs.wz)

        if self.cont_steps >= 1:
            reward, done = compute_reward(self.previous_obs, inputs)

            self.episode_rewards.append(reward)

            masks = torch.FloatTensor([0 if done else 1])
            self.memory.insert(obs, self.previous_action, self.previous_action_log_prob, self.previous_value, reward, masks)

            with torch.no_grad():
                next_value = self.agents.get_value(inputs).detach()

            self.memory.compute_returns(next_value, self.info_dict['use_gae'], self.info_dict['gamma'],
                                        self.info_dict['gae_lambda'])

            value_loss, action_loss, dist_entropy = self.ppo.update(self.memory)

            self.memory.after_update()

            if self.cont_steps % self.info_dict['log_interval'] == 0:
                end = time.time()
                print(
                    "Num timesteps {}, FPS {} \n Last {} training episodes: mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}\n"
                    .format(self.cont_steps, int(self.cont_steps / (end - self.start)),
                            len(self.episode_rewards), np.mean(self.episode_rewards),
                            np.median(self.episode_rewards), np.min(self.episode_rewards),
                            np.max(self.episode_rewards), dist_entropy, value_loss,
                            action_loss))

        with torch.no_grad():
            self.previous_value, action, self.previous_action_log_prob = \
                self.agents.act(torch.FloatTensor(inputs))

            # Obser reward and next obs
            action_msg = Float32MultiArray()
            action_msg[0] = action[0]
            action_msg[1] = action[1]
            action_msg[2] = action[2]
            self.vehicle_odometry_subscriber.publish(action_msg)

            self.previous_obs = inputs
            self.previous_action = action

        self.cont_steps += 1

        if self.cont_steps % self.info_dict['num-steps'] == 0:
            if self.cont_steps % self.info_dict['train_freq'] == 0:
                pass

        if self.cont_steps >= self.info_dict['num-env-steps']:
            self.memory.to_csv("data/memory.csv")
            print("Training ended, total time: ", time.time()-self.start)
            rclpy.shutdown()

    def timestamp_callback(self, msg):
        self.timestamp_ = msg.timestamp


def compute_reward(previous_obs, obs):
    return np.sum(obs-previous_obs)


if __name__ == '__main__':
    rclpy.init(args=None)
    m_node = rclpy.create_node('gs_node')
    gsNode = AgentNode(m_node)
    rclpy.spin(m_node)
