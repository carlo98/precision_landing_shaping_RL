#!/usr/bin/env python3
# Python dep
import yaml
import numpy as np
import time
import torch
import threading
import sys

# ROS
import rclpy

sys.path.append("/src/shared/ros_packages/px4_ros_extended/src_py")

# DDPG
from DDPG.ddpg import DDPG
from DDPG.memory import Memory

from env_wrapper import EnvWrapperNode


class AgentNode:
    def __init__(self, node):

        with open('/src/shared/ros_packages/px4_ros_extended/src_py/params.yaml') as info:
            self.info_dict = yaml.load(info, Loader=yaml.SafeLoader)

        torch.manual_seed(self.info_dict['seed'])
        torch.cuda.manual_seed_all(self.info_dict['seed'])

        self.env = EnvWrapperNode(node)
        self.memory = Memory(self.info_dict['max_memory_len'])
        self.ddpg = DDPG(self.info_dict['obs_shape'], self.info_dict['action_space'], 1,  self.memory)

        self.start = time.time()

        self.cont_steps = 0
        self.previous_obs = np.zeros(self.info_dict['obs_shape'])
        self.previous_action = np.zeros(self.info_dict['action_space'])
        self.previous_action_log_prob = np.zeros(self.info_dict['action_space'])
        self.previous_value = 0.0

    def train(self):
    
        while self.env.reset:  # Waiting for env to stop resetting
            self.previous_obs = inputs = self.env.state
            
        self.env.play_env()  # Start landing listening

        while self.cont_steps <= self.info_dict['num-env-steps']:
            with torch.no_grad():
                action = self.ddpg.get_exploration_action(inputs)
            
            inputs, reward, done = self.env.act(action)

            self.memory.add(self.previous_obs, action, reward, inputs)

            if self.cont_steps % self.info_dict['num-steps'] == 0:
                print("End episode")
                self.env.reset_env()
                if self.cont_steps % self.info_dict['train_freq'] == 0:
                    self.ddpg.optimize()
                    
                self.env.play_env()  # Restart landing listening, after training
            
            if done:
                print("Done")
                self.env.reset_env()
                self.env.play_env()  # Restart landing listening, after done
                
            self.previous_obs = inputs
            inputs = self.env.state
            self.cont_steps += 1

        self.ddpg.save_models(self.cont_steps)

 
def spin_thread(node):
    rclpy.spin(node)


if __name__ == '__main__':
    rclpy.init(args=None)
    m_node = rclpy.create_node('agent_node')
    gsNode = AgentNode(m_node)
    x = threading.Thread(target=spin_thread, args=(m_node,))
    x.start()
    gsNode.train()
    x.join()
    
    
