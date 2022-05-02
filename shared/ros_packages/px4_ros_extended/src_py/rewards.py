#!/usr/bin/env python3
import numpy as np


class Reward:
    def __init__(self, max_height, max_side):
        self.min_reward = -1
        # Weights for pos, velocity, action, 2(3) x single action, coef pos z
        self.coeffs = np.array([-100, -10, -1, 10, 10, 10])
        self.previous_shaping = 0.0
        
        self.max_height = max_height
        self.max_side = max_side

    def init_shaping(self, obs):
        self.previous_shaping = self.coeffs[0] * np.sqrt(obs[0] ** 2 + obs[1] ** 2) + \
                                self.coeffs[1] * np.sqrt(obs[3] ** 2 + obs[4] ** 2)

    def get_reward(self, obs, norm_obs, action, landed, eps_pos_xy, eps_vel_xy):
        done = False
        c = 0.0
        
        # Reward for position in z-axes should avoid hovering at z=0 if terrain is lower
        landed = landed or norm_obs[2] <= 0.0
        
        # Slowly Landed in objective
        if landed and np.abs(obs[3]) <= eps_vel_xy and np.abs(obs[4]) <= eps_vel_xy \
                and np.abs(obs[0]) <= eps_pos_xy and np.abs(obs[1]) <= eps_pos_xy:
            print("Slowly Landed in obj.")
            c = 1.0
            done = True

        # Outside of area
        elif np.abs(obs[2]) > self.max_height or np.abs(obs[0]) > self.max_side \
                or np.abs(obs[1]) > self.max_side:
            print("Outside of area.")
            done = True

        # Landed in wrong place
        elif landed and (np.abs(obs[0]) > eps_pos_xy or np.abs(obs[1]) > eps_pos_xy):
            print("Landed in wrong place.")
            done = True
            
        # Landed in obj
        elif landed and np.abs(obs[0]) <= eps_pos_xy and np.abs(obs[1]) <= eps_pos_xy:
            print("Fast Landing in obj")
            c = 1.0
            done = True

        if len(action) == 2:  # Predicting only vx and vy
            shaping = self.coeffs[0] * np.sqrt(norm_obs[0] ** 2 + norm_obs[1] ** 2) + \
                      self.coeffs[1] * np.sqrt(norm_obs[3] ** 2 + norm_obs[4] ** 2) + \
                      self.coeffs[2] * np.sqrt(action[0] ** 2 + action[1] ** 2) + \
                      self.coeffs[3] * c * (1 - np.abs(action[0])) + \
                      self.coeffs[4] * c * (1 - np.abs(action[1]))
        elif len(action) == 3:  # Predicting vx, vy, vz
            # Reward for position in z-axes should avoid hovering at z=0 if terrain is lower
            if norm_obs[2] > 0.0:
                shaping = self.coeffs[0] * np.sqrt(norm_obs[0] ** 2 + norm_obs[1] ** 2 + norm_obs[2] ** 2)
            else:
                shaping = self.coeffs[0] * np.sqrt(norm_obs[0] ** 2 + norm_obs[1] ** 2)
                
            shaping += self.coeffs[1] * np.sqrt(norm_obs[3] ** 2 + norm_obs[4] ** 2 + norm_obs[5] ** 2) + \
                       self.coeffs[2] * np.sqrt(action[0] ** 2 + action[1] ** 2 + action[2] ** 2) + \
                       self.coeffs[3] * c * (1 - np.abs(action[0])) + \
                       self.coeffs[4] * c * (1 - np.abs(action[1])) + \
                       self.coeffs[5] * c * (1 - np.abs(action[2]))

        reward = shaping - self.previous_shaping
        self.previous_shaping = shaping
        return reward, done

