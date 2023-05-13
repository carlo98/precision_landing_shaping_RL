import torch
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

    def sample(self, count, batch_size):
        """
        Samples a random batch from the replay memory buffer.
        :param count: mem_to_sample
        :param batch_size: batch size
        :return: generator of batches
        """
        count = min(count, self.length)
        indices = random.sample(range(self.length), count)

        for i in range(0, count, batch_size):
            batch_indices = indices[i:i+batch_size]
            batch = [self.buffer[idx] for idx in batch_indices]

            s_arr = torch.tensor(np.array([arr[0] for arr in batch]), dtype=torch.float32)
            a_arr = torch.tensor(np.array([arr[1] for arr in batch]), dtype=torch.float32)
            r_arr = torch.tensor(np.array([arr[2] for arr in batch]), dtype=torch.float32)
            s1_arr = torch.tensor(np.array([arr[3] for arr in batch]), dtype=torch.float32)
            done_arr = torch.tensor(np.array([arr[4] for arr in batch]), dtype=torch.int32)

            yield s_arr, a_arr, r_arr, s1_arr, done_arr

    def len(self):
        return self.length
        
    def log(self):
        filename = '/log_' + str(self.id_file) + ".pkl"
        with open(self.path_logs+filename, "wb") as pkl_f:
            pickle.dump([self.acc_rewards_train, self.acc_rewards_test], pkl_f)

    def add(self, s, a, r, s1, done, le):
        """
        Adds a particular transaction in the memory buffer.
        :param s: current state
        :param a: action taken
        :param r: reward received
        :param s1: next state
        :param done: done flag
        :param le: log episode id (int)
        """
        transition = (s, a, r, s1, done, le)
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
