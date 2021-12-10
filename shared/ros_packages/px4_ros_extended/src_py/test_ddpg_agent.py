#!/usr/bin/env python3
# Python dep
import yaml
import numpy as np
import torch
import threading
import sys
import pickle
import os
import time
import argparse
from datetime import datetime

# ROS
import rclpy

sys.path.append("/src/shared/ros_packages/px4_ros_extended/src_py")

# DDPG
from DDPG.ddpg import DDPG
from DDPG.memory import Memory
from env_wrapper import EnvWrapperNode


class AgentNode:
    def __init__(self, node, run_id, model_name, obs_shape):

        with open('/src/shared/ros_packages/px4_ros_extended/src_py/params.yaml') as info:
            self.info_dict = yaml.load(info, Loader=yaml.SafeLoader)

        torch.manual_seed(self.info_dict['seed'])
        torch.cuda.manual_seed_all(self.info_dict['seed'])

        self.env = EnvWrapperNode(node, obs_shape, self.info_dict['action_space'], self.info_dict['max_height'], self.info_dict['max_side'],
                                  self.info_dict['max_vel_z'], self.info_dict['max_vel_xy'], self.info_dict['eps_pos_xy'])
        self.memory = Memory(self.info_dict['max_memory_len'])
        self.ddpg = DDPG(obs_shape, self.info_dict['action_space'], self.memory, model_name,
                         lr_actor=self.info_dict['lr_actor'], lr_critic=self.info_dict['lr_critic'], gamma=self.info_dict['gamma'],
                         tau=self.info_dict['tau'], batch_size=self.info_dict['batch_size'],
                         epochs=self.info_dict['epochs'])

        self.ddpg.load_models(run_id, best=True)

        self.run_id = run_id
        base_dir = "/src/shared/test_logs"
        if not os.path.isdir(base_dir):
            os.mkdir(base_dir)

        self.path_logs = os.path.join(base_dir, str(self.run_id))
        if not os.path.isdir(self.path_logs):
            os.mkdir(self.path_logs)

    def run(self):

        log_positions = {'x': [], 'y': [], 'z': []}  # Saving position from target in x,y,z-axis
        log_velocities = {'vx': [], 'vy': []}  # Saving relative velocity to target in x,y-axis (actions)
        if self.info_dict['action_space'] == 3:  # Predicting also vz
            log_velocities['vz'] = []
        cont_steps = 0
        episode_tot_reward = 0
        while self.env.reset:  # Waiting for env to stop resetting
            pass

        inputs = self.env.play_env()  # Start landing listening in src_cpp/env.cpp
        start_time = time.time()  # Used for plots
        log_positions['x'].append(inputs[0])
        log_positions['y'].append(inputs[1])
        log_positions['z'].append(inputs[2])

        while True:
            normalized_input = self.normalize_input(np.copy(inputs))
            with torch.no_grad():
                action = self.ddpg.get_exploitation_action(normalized_input)

            log_velocities['vx'].append(action[0]*self.info_dict['max_vel_xy'])
            log_velocities['vy'].append(action[1]*self.info_dict['max_vel_xy'])
            if self.info_dict['action_space'] == 3:  # Predicting also vz
                log_velocities['vz'].append(action[2]*self.info_dict['max_vel_z'])

            inputs, reward, done = self.env.act(action, self.normalize_input)
            log_positions['x'].append(inputs[0])
            log_positions['y'].append(inputs[1])
            log_positions['z'].append(inputs[2])
            episode_tot_reward += reward

            if (cont_steps % self.info_dict['num-steps'] == 0 and cont_steps > 0) or done:
                if done:
                    print("Done")
                else:
                    print("End episode")
                print("Position x: " + str(-inputs[0]) + " y: " + str(-inputs[1]) + " z: " + str(-inputs[2]))
                print("Acc reward: " + str(episode_tot_reward))

                self.env.reset_env()
                    
                episode_tot_reward = 0.0
                self.log(log_positions, log_velocities, time.time()-start_time)

                inputs = self.env.play_env()  # Restart landing listening, after training
                log_positions = {'x': [inputs[0]], 'y': [inputs[1]], 'z': [inputs[2]]}
                log_velocities = {'vx': [], 'vy': []}
                if self.info_dict['action_space'] == 3:  # Predicting also vz
                    log_velocities['vz'] = []
                start_time = time.time()

                print("\nNew episode started\n")

            cont_steps += 1

    def normalize_input(self, inputs):
        inputs[:2] /= self.info_dict['max_side']
        inputs[2] /= self.info_dict['max_height']
        inputs[3:] /= self.info_dict['max_vel_xy']
        return inputs

    def log(self, positions, velocities, passed_time):
        filename = '/test_log_' + str(self.run_id) + "_" + str(datetime.now()).split(".")[0] + ".pkl"
        with open(self.path_logs+filename, "wb") as pkl_f:
            pickle.dump([positions, velocities, passed_time], pkl_f)

 
def spin_thread(node):
    rclpy.spin(node)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('run_id', type=int, help="Id of run to be loaded")
    parser.add_argument('model', type=str, help="Name of the model used. Either 'paper' or 'small'")
    parser.add_argument('obs_shape', type=int, help="Shape of observation. Either 5 or 6")
    args = parser.parse_args()

    print("Starting Agent and Env Wrapper")
    rclpy.init(args=None)
    m_node = rclpy.create_node('agent_node')
    gsNode = AgentNode(m_node, args.run_id, args.model, args.obs_shape)
    x = threading.Thread(target=spin_thread, args=(m_node,))
    x.start()
    gsNode.run()
    x.join()
