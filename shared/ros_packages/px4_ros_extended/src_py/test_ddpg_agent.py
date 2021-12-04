#!/usr/bin/env python3
# Python dep
import yaml
import numpy as np
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

        self.env = EnvWrapperNode(node, self.info_dict['obs_shape'], self.info_dict['max_height'], self.info_dict['max_side'],
                                  self.info_dict['max_vel_z'], self.info_dict['max_vel_xy'])
        self.memory = Memory(self.info_dict['max_memory_len'])
        self.ddpg = DDPG(self.info_dict['obs_shape'], self.info_dict['action_space'], self.memory,
                         lr_actor=self.info_dict['lr_actor'], lr_critic=self.info_dict['lr_critic'], gamma=self.info_dict['gamma'],
                         tau=self.info_dict['tau'], batch_size=self.info_dict['batch_size'],
                         epochs=self.info_dict['epochs'])

        self.ddpg.load_models(1, best=True)

    def run(self):

        cont_steps = 0
        episode_tot_reward = 0
        while self.env.reset:  # Waiting for env to stop resetting
            pass

        inputs = self.env.play_env()  # Start landing listening in src_cpp/env.cpp

        while True:
            normalized_input = self.normalize_input(np.copy(inputs))
            with torch.no_grad():
                action = self.ddpg.get_exploitation_action(normalized_input)
            
            inputs, reward, done = self.env.act(action, self.normalize_input)
            episode_tot_reward += reward

            normalized_input = self.normalize_input(np.copy(inputs))

            if (cont_steps % self.info_dict['num-steps'] == 0 and cont_steps > 0) or done:
                if done:
                    print("Done")
                else:
                    print("End episode")
                print("Position x: " + str(-inputs[0]) + " y: " + str(-inputs[1]) + " z: " + str(-inputs[2]))
                print("Acc reward: " + str(episode_tot_reward))

                self.env.reset_env()
                    
                episode_tot_reward = 0.0

                inputs = self.env.play_env()  # Restart landing listening, after training
                print("\nNew episode started\n")

            cont_steps += 1

    def normalize_input(self, inputs):
        inputs[:2] /= self.info_dict['max_side']
        inputs[2] /= self.info_dict['max_height']
        inputs[3:] /= self.info_dict['max_vel_xy']
        return inputs

 
def spin_thread(node):
    rclpy.spin(node)


if __name__ == '__main__':
    print("Starting Agent and Env Wrapper")
    rclpy.init(args=None)
    m_node = rclpy.create_node('agent_node')
    gsNode = AgentNode(m_node)
    x = threading.Thread(target=spin_thread, args=(m_node,))
    x.start()
    gsNode.run()
    x.join()
