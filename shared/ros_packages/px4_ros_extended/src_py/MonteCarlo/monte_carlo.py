"""
This class performs a Monte Carlo Search at different heights starting from a given threshold.
"""
import numpy as np
import pickle


class MonteCarlo:
    def __init__(self, id_file, path_logs, action_dim, start_height=100, steps_dist=5, num_samples=100, scale=0.1):
        """
        Initialize MonteCarlo object
        :param id_file: Used to log actions, first part of each file name
        :param path_logs: Folder in which to log actions
        :param action_dim: action's dimension
        :param start_height: Height at which to start Monte Carlo Simulation (centimeters)
        :param steps_dist: Perform sampling every 'steps_dist' centimeters
        :param num_samples: Number of samples for each step, i.e. for start_height/steps_dist times
        :param scale: standard deviation for gaussian distribution
        :return:
        """
        self.id_file = id_file
        self.path_logs = path_logs
        self.start_height = start_height
        self.steps_dist = steps_dist
        self.num_samples = num_samples
        self.scale = scale
        self.action_dim = action_dim

        self.steps = int(self.start_height/self.steps_dist)
        self.action_log = np.zeros((self.steps, self.num_samples, self.action_dim))

        self.curr_pos = -1
        self.curr_samples = 0
        self.episodes = 0

    def generate_samples(self, model_action):
        """
        :param model_action: Action predicted by model, used as mean when sampling (vx, vy[, vz])
        """
        self.curr_pos += 1
        for i in range(self.num_samples):
            action = np.random.normal(loc=model_action, scale=self.scale)
            self.action_log[self.curr_pos][i] = action
        self.curr_samples = 0

    def sample(self):
        """
        Returns next action from given position
        """
        action = self.action_log[self.curr_pos][self.curr_samples]
        self.curr_samples += 1
        return action

    def reset(self):
        """
        Resets data, starting a new episode
        """
        self._log()
        self.curr_pos = 0
        self.curr_samples = 0
        self.action_log = np.zeros((self.steps, self.num_samples, self.action_dim))
        self.episodes += 1

    def _log(self):
        filename = '/log_' + str(self.id_file) + "_" + str(self.episodes) + ".pkl"
        with open(self.path_logs+filename, "wb") as pkl_f:
            pickle.dump(self.action_log, pkl_f)
