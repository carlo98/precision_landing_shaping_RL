#!/usr/bin/env python3
# Python dep
import yaml
import numpy as np
from collections import deque
import time
import torch

# DDPG
from DDPG.ddpg import DDPG
from DDPG.memory import Memory

from env_wrapper import EnvWrapperNode


class Agent:
    def __init__(self):

        with open('params.yaml') as info:
            self.info_dict = yaml.load(info)

        torch.manual_seed(self.info_dict['seed'])
        torch.cuda.manual_seed_all(self.info_dict['seed'])

        self.memory = Memory(self.info_dict['max_memory_len'])

        self.ddpg = DDPG(self.info_dict['obs_shape'], self.info_dict['action_space'], 1,  self.memory)

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
            self.memory.add(self.previous_obs, self.previous_action, reward, inputs)

            if self.cont_steps % self.info_dict['num-steps'] == 0:
                if self.cont_steps % self.info_dict['train_freq'] == 0:
                    self.ddpg.optimize()

        with torch.no_grad():
            action = self.ddpg.get_exploration_action(inputs)

            # Obser reward and next obs
            self.vehicle_odometry_subscriber.publish(action_msg)

            self.previous_obs = inputs
            self.previous_action = action

        self.cont_steps += 1

        if self.cont_steps >= self.info_dict['num-env-steps']:
            self.ddpg.save_models(self.cont_steps)

    def timestamp_callback(self, msg):
        self.timestamp_ = msg.timestamp


def compute_reward(previous_obs, obs):
    return np.sum(obs-previous_obs)


if __name__ == '__main__':
    rclpy.init(args=None)
    m_node = rclpy.create_node('gs_node')
    gsNode = AgentNode(m_node)
    rclpy.spin(m_node)
