#!/usr/bin/env python3
# Python dep
import yaml
import numpy as np
import time
import torch
import threading
import sys
import subprocess

# ROS
import rclpy

sys.path.append("/src/shared/ros_packages/px4_ros_extended/src_py")

# DDPG
from DDPG.ddpg import DDPG
from DDPG.memory import Memory

# Monte Carlo
from MonteCarlo.monte_carlo import MonteCarlo

# Environment
from env_wrapper import EnvWrapperNode


class AgentNode:
    def __init__(self, node):
        with open('/src/shared/ros_packages/px4_ros_extended/src_py/params.yaml') as info:
            self.info_dict = yaml.load(info, Loader=yaml.SafeLoader)

        torch.manual_seed(self.info_dict['seed'])
        torch.cuda.manual_seed_all(self.info_dict['seed'])

        self.env = EnvWrapperNode(node, self.info_dict['obs_shape'], self.info_dict['action_space'],
                                  self.info_dict['max_height'], self.info_dict['max_side'], self.info_dict['max_vel_z'],
                                  self.info_dict['max_vel_xy'], self.info_dict['eps_pos_xy'])
        self.memory = Memory(self.info_dict['max_memory_len'])
        self.ddpg = DDPG(self.info_dict['obs_shape'], self.info_dict['action_space'], self.memory,
                         self.info_dict['model'], lr_actor=self.info_dict['lr_actor'],
                         lr_critic=self.info_dict['lr_critic'], gamma=self.info_dict['gamma'],
                         tau=self.info_dict['tau'], batch_size=self.info_dict['batch_size'],
                         epochs=self.info_dict['epochs'])
        self.mc = MonteCarlo(self.memory.get_id_file(), "/src/shared/mc_logs", self.info_dict['action_space'],
                             start_height=self.info_dict['start_height'], steps_dist=self.info_dict['steps_dist'],
                             num_samples=self.info_dict['num_samples'], scale=self.info_dict['scale'])

        self.eval_noise = self.info_dict['eval_noise']
        self.start_time = time.time()

    def train(self):
        episode_tot_reward = 0.0
        episode_num = 0
        cont_test = 0
        episode_steps = 0
        cont_steps = 0
        best_eval_reward = 0.0
        mc_step_id = 0

        while self.env.reset:  # Waiting for env to stop resetting
            pass

        inputs = self.env.play_env()  # Start landing listening in src_cpp/env.cpp
        start_time_episode = time.time()

        while episode_num < self.info_dict['num_env_episodes']:
            evaluating = episode_num != 0 and (episode_num % self.info_dict['evaluate_freq'] == 0 or
                                               0 < cont_test < self.info_dict['evaluate_ep'])

            episode_steps += 1
            if self.info_dict["use_mc"] == "False" or \
                    (self.info_dict["use_mc"] == "True" and inputs[2] > self.mc.get_start_height()):
                with torch.no_grad():
                    if evaluating:  # Evaluate model
                        # During evaluation adding noise to the state
                        inputs = np.random.normal(loc=inputs, scale=self.eval_noise)
                        normalized_input = self.normalize_input(np.copy(inputs))
                        action = self.ddpg.get_exploitation_action(normalized_input)[0]
                    else:
                        normalized_input = self.normalize_input(np.copy(inputs))
                        action = self.ddpg.get_exploration_action(normalized_input, cont_steps)[0]
            elif self.info_dict["use_mc"] == "True" and inputs[2] <= self.mc.get_start_height():
                lower_bound = self.mc.get_start_height() - (mc_step_id+1)*self.mc.get_steps_dist()
                upper_bound = self.mc.get_start_height() - mc_step_id*self.mc.get_steps_dist()
                if not (lower_bound <= inputs[2] <= upper_bound):
                    mc_step_id += 1
                    normalized_input = self.normalize_input(np.copy(inputs))
                    model_action = self.ddpg.get_exploitation_action(normalized_input)[0]
                    self.mc.generate_samples(model_action)

                action = self.mc.sample()

            inputs, reward, done = self.env.act(action, self.normalize_input)

            previous_obs = np.copy(normalized_input)
            normalized_input = self.normalize_input(np.copy(inputs))

            # Saving episode in memory if not using Monte Carlo
            if episode_steps > 1 and \
                    not (self.info_dict["use_mc"] == "True" and inputs[2] <= self.mc.get_start_height()):
                episode_tot_reward += reward
                self.memory.add(previous_obs, action, reward, normalized_input, episode_num)
            elif episode_steps > 1 and self.info_dict["use_mc"] == "True" and inputs[2] <= self.mc.get_start_height():
                pass  # TODO: Monte Carlo rewards in memory

            if (cont_steps % self.info_dict['num-steps'] == 0 and cont_steps > 0 and episode_steps > 1) or done:
                # TODO: reset position of agent for Monte Carlo
                if done:
                    print("Done " + str(episode_num))
                else:
                    print("End episode " + str(episode_num))
                if evaluating:
                    cont_test += 1
                    print("Evaluation episode " + str(cont_test))
                    if cont_test >= self.info_dict['evaluate_ep']:
                        cont_test = 0
                        mean_reward_eval = np.mean(self.memory.acc_rewards_test[-self.info_dict['evaluate_ep']:])

                        # Saving best model based on mean evaluation reward of group
                        if mean_reward_eval > best_eval_reward:
                            print("Saving best model.")
                            self.ddpg.save_models(self.memory.get_id_file(), best=True)
                            best_eval_reward = mean_reward_eval

                print("Position x: " + str(-inputs[0]) + " y: " + str(-inputs[1]) + " z: " + str(-inputs[2]))
                print("Acc reward: " + str(episode_tot_reward) + " Time: " + str(time.time()-start_time_episode))

                self.memory.add_acc_reward(episode_tot_reward, evaluating)
                self.env.reset_env()
                if episode_num % self.info_dict['train_freq'] == 0 and self.memory.len() >= self.info_dict['min_mem'] \
                        and not evaluating:  # Do not train during evaluation episodes
                    print("Training...")
                    self.ddpg.optimize(self.info_dict['mem_to_use'])
                    print("Training ended")

                if episode_num != 0 and episode_num % self.info_dict['log_interval_episodes'] == 0:
                    print("Saving rewards in file.")
                    self.memory.log()
                    print("Saving model.")
                    self.ddpg.save_models(self.memory.get_id_file(), best=False)

                episode_tot_reward = 0.0
                episode_num += 1
                episode_steps = 0
                if self.info_dict["use_mc"] == "True":
                    self.mc.reset()  # Resetting Monte Carlo

                inputs = self.env.play_env()  # Restart landing listening, after training
                print("\nNew episode started\n")
                start_time_episode = time.time()

            cont_steps += 1

        print("Saving rewards in file.")
        self.memory.log()
        print("Saving model...\nTotal time: ", time.time()-self.start_time)
        self.ddpg.save_models(self.memory.get_id_file())

        self.env.shutdown_gazebo()

    def normalize_input(self, inputs):
        inputs[:2] /= self.info_dict['max_side']
        inputs[2] /= self.info_dict['max_height']
        inputs[3:] /= self.info_dict['max_vel_xy']
        return inputs


def spin_thread(node):
    rclpy.spin(node)


if __name__ == '__main__':
    print("Starting micrortps agent")

    micrortps_agent = subprocess.Popen(["micrortps_agent", "-t", "UDP"])
    time.sleep(2)

    print("Starting Agent and Env Wrapper")
    rclpy.init(args=None)
    m_node = rclpy.create_node('agent_node')
    gsNode = AgentNode(m_node)
    x = threading.Thread(target=spin_thread, args=(m_node,))
    x.start()
    gsNode.train()
    rclpy.shutdown()  # Closing Env Wrapper Node
    x.join()
    micrortps_agent.kill()  # Closing micrortps agent
