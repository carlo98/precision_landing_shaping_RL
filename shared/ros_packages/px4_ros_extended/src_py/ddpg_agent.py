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
        self.ddpg = DDPG(self.info_dict['obs_shape'], self.info_dict['action_space'], self.memory,
                         lr=self.info_dict['lr'], gamma=self.info_dict['gamma'],
                         tau=self.info_dict['tau'], batch_size=self.info_dict['batch_size'],
                         epochs=self.info_dict['epochs'])
                         
        self.num_log_episodes = self.info_dict['num-steps'] * self.info_dict['log_interval_episodes']
        
        self.cont_steps = 0
        self.previous_obs = np.zeros(self.info_dict['obs_shape'])
        self.previous_action = np.zeros(self.info_dict['action_space'])
        self.previous_action_log_prob = np.zeros(self.info_dict['action_space'])
        self.previous_value = 0.0

        self.start_time = time.time()

    def train(self):
    
        episode_steps = 0
        episode_tot_reward = 0.0

        self.previous_obs = inputs = self.env.state
        while self.env.reset:  # Waiting for env to stop resetting
            self.previous_obs = inputs = self.env.state
            
        self.env.play_env()  # Start landing listening
        start_time_episode = time.time()

        while self.cont_steps <= self.info_dict['num-env-steps']:
            with torch.no_grad():
                action = self.ddpg.get_exploration_action(inputs)
            
            inputs, reward, done = self.env.act(action)
            episode_tot_reward += reward

            self.memory.add(self.previous_obs, action, reward, inputs, int(self.cont_steps / self.info_dict['num-steps']))

            if (self.cont_steps % self.info_dict['num-steps'] == 0 and self.cont_steps > 0) or done:
                if done:
                    print("Done")
                else:
                    print("End episode " + str(int(self.cont_steps / self.info_dict['num-steps'])))
                print("Position x: " + str(-inputs[0]) + " y: " + str(-inputs[1]) + " z: " + str(-inputs[2]))
                print("Mean reward: " + str(episode_tot_reward / episode_steps) + " Time: " + str(time.time()-start_time_episode))
                self.env.reset_env()
                if self.cont_steps % self.info_dict['train_freq'] == 0:
                    print("Training...")
                    self.ddpg.optimize()
                    print("Training ended")
                    
                if self.cont_steps % self.num_log_episodes == 0:
                    print("Saving in memory file.")
                    self.memory.log()
                print()
                    
                self.env.play_env()  # Restart landing listening, after training
                start_time_episode = time.time()
                episode_steps = 0
                episode_tot_reward = 0.0
                
            self.previous_obs = inputs
            inputs = self.env.state
            self.cont_steps += 1
            episode_steps += 1

        print("Saving model...\nTotal time: ", time.time()-self.start_time)
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
    
    
