import numpy as np
import random
import pickle
import re
from collections import deque
import os


class Memory:
    def __init__(self, size):
        self.buffer = deque(maxlen=size)
        self.acc_rewards_train = []
        self.acc_rewards_test = []
        self.maxSize = size
        self.length = 0
        
        self.path_logs = "/src/shared/logs"
        if not os.path.isdir(self.path_logs):
            os.mkdir(self.path_logs)
            self.id_file = 0
        else:
            max_folder = -1
            regex_folder_name = re.compile("^log_[0-9]*.pkl$")
            for folder in os.listdir(self.path_logs):
                if regex_folder_name.match(folder) is not None:
                    end = folder.split("_")[1]
                    number = int(end.split(".")[0])
                    if number > max_folder:
                        max_folder = number
            self.id_file = max_folder + 1

    def sample(self, count):
        """
        samples a random batch from the replay memory buffer
        :param count: batch size
        :return: batch (numpy array)
        """
        count = min(count, self.length)
        batch = random.sample(self.buffer, count)

        s_arr = np.float32([arr[0] for arr in batch])
        a_arr = np.float32([arr[1] for arr in batch])
        r_arr = np.float32([arr[2] for arr in batch])
        s1_arr = np.float32([arr[3] for arr in batch])

        return s_arr, a_arr, r_arr, s1_arr

    def len(self):
        return self.length
        
    def log(self):
        filename = '/log_' + str(self.id_file) + ".pkl"
        with open(self.path_logs+filename, "wb") as pkl_f:
            pickle.dump([self.acc_rewards_train, self.acc_rewards_test], pkl_f)

    def add(self, s, a, r, s1, le):
        """
        adds a particular transaction in the memory buffer
        :param s: current state
        :param a: action taken
        :param r: reward received
        :param s1: next state
        :param le: log episode id (int)
        :return:
        """
        transition = (s, a, r, s1, le)
        if self.length >= self.maxSize:
            self.length = self.maxSize
        else:
            self.length += 1
        self.buffer.append(transition)

    def add_acc_reward(self, acc_r, evaluating):
        if evaluating:
            self.acc_rewards_test.append(acc_r)
        else:
            self.acc_rewards_train.append(acc_r)

