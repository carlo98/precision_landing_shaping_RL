#!/usr/bin/env python3
import numpy as np


class Reward:
    def __init__(self):
        self.min_reward = -1
        # Weights for pos, velocity, angular velocity, action, 3 x single action
        self.coeffs = np.array([-100, -10, -10, -1, 10, 10, 10])
        self.stop_reward = 100
        self.previous_shaping = 0.0
        
        self.max_height = 5.0
        self.max_side = 5.0

    def init_shaping(self, obs):
        self.previous_shaping = self.coeffs[0] * np.sqrt(obs[0] ** 2 + obs[1] ** 2 + obs[2] ** 2) + \
                                self.coeffs[1] * np.sqrt(obs[3] ** 2 + obs[4] ** 2 + obs[5] ** 2)

    def get_reward(self, obs, action, eps_pos_z, eps_vel_z, eps_vel_xy):
        done = False
        if np.abs(obs[2]) <= eps_pos_z and np.abs(obs[5]) <= eps_vel_z \
                and np.abs(obs[3]) <= eps_vel_xy and np.abs(obs[4]) <= eps_vel_xy:
            done = True
            
        if np.abs(obs[2]) > self.max_height or np.abs(obs[0]) > self.max_side \
                or np.abs(obs[1]) > self.max_side:
            done = True

        shaping = self.coeffs[0] * np.sqrt(obs[0] ** 2 + obs[1] ** 2 + obs[2] ** 2) + \
                  self.coeffs[1] * np.sqrt(obs[3] ** 2 + obs[4] ** 2 + obs[5] ** 2) + \
                  self.coeffs[2] * np.sqrt(action[0] ** 2 + action[1] ** 2 + action[2] ** 2) + \
                  self.coeffs[3] * self.stop_reward * (1 - np.abs(action[0])) + \
                  self.coeffs[4] * self.stop_reward * (1 - np.abs(action[1])) + \
                  self.coeffs[5] * self.stop_reward * (1 - np.abs(action[2]))

        reward = shaping - self.previous_shaping
        self.previous_shaping = shaping
        return reward, done

    def get_reward_1(self, obs, action, collision, eps_pos_z, eps_vel_z, eps_vel_xy):
        if collision:
            return self.min_reward, True

        done = False
        if obs[2] <= eps_pos_z and obs[5] <= eps_vel_z \
                and obs[3] <= eps_vel_xy and obs[4] <= eps_vel_xy:
            done = True

        shaping = self.coeffs[0] * np.sqrt(obs[0] ** 2 + obs[1] ** 2 + obs[2] ** 2) + \
                  self.coeffs[1] * np.sqrt(obs[3] ** 2 + obs[4] ** 2 + obs[5] ** 2) + \
                  self.coeffs[2] * np.sqrt(action[0] ** 2 + action[1] ** 2 + action[2] ** 2) + \
                  self.coeffs[3] * self.stop_reward * (1 - np.abs(action[0])) + \
                  self.coeffs[4] * self.stop_reward * (1 - np.abs(action[1])) + \
                  self.coeffs[5] * self.stop_reward * (1 - np.abs(action[2]))

        reward = shaping - self.previous_shaping
        self.previous_shaping = shaping
        return reward, done
