#!/usr/bin/env python3
import numpy as np


class Reward:
    def __init__(self, max_height, max_side):
        self.min_reward = -1
        # Weights for pos, velocity, action, 2|3 x single action
        self.coeffs = np.array([-10, -1, -0.1, 1, 1])  # x10 paper
        self.coeffs_1 = np.array([-10, -1, -0.1, 1, 1, 1])
        self.stop_reward_paper = 100.0
        self.stop_reward = 10.0
        self.previous_shaping = 0.0
        
        self.max_height = max_height
        self.max_side = max_side

    def init_shaping(self, obs):
        self.previous_shaping = self.coeffs[0] * np.sqrt(obs[0] ** 2 + obs[1] ** 2 + obs[2] ** 2) + \
                                self.coeffs[1] * np.sqrt(obs[3] ** 2 + obs[4] ** 2 + obs[5] ** 2)

    def get_reward(self, obs, action, eps_pos_z, eps_pos_xy, eps_vel_xy):
        done = False

        # Landed in objective
        if np.abs(obs[2]) <= eps_pos_z and np.abs(obs[3]) <= eps_vel_xy and np.abs(obs[4]) <= eps_vel_xy \
                and np.abs(obs[0]) <= eps_pos_xy and np.abs(obs[1]) <= eps_pos_xy:
            print("Landed in obj.")
            done = True

        # Outside of area
        if np.abs(obs[2]) > self.max_height or np.abs(obs[0]) > self.max_side \
                or np.abs(obs[1]) > self.max_side:
            print("Outside of area.")
            done = True

        # Landed in wrong place
        if (np.abs(obs[2]) <= eps_pos_z and np.abs(obs[3]) <= eps_vel_xy and np.abs(obs[4]) <= eps_vel_xy) \
                and (np.abs(obs[0]) > eps_pos_xy or np.abs(obs[1]) > eps_pos_xy):
            print("Landed in wrong place.")
            done = True

        shaping = self.coeffs[0] * np.sqrt(obs[0] ** 2 + obs[1] ** 2) + \
                  self.coeffs[1] * np.sqrt(obs[3] ** 2 + obs[4] ** 2) + \
                  self.coeffs[2] * np.sqrt(action[0] ** 2 + action[1] ** 2) + \
                  self.coeffs[3] * self.stop_reward * (1 - np.abs(action[0])) + \
                  self.coeffs[4] * self.stop_reward * (1 - np.abs(action[1]))

        reward = shaping - self.previous_shaping
        self.previous_shaping = shaping
        return reward, done
