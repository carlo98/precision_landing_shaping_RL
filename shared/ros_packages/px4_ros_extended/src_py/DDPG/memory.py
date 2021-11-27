import numpy as np
import random
import pandas as pd
from collections import deque
from datetime import datetime
import os


class Memory:
    def __init__(self, size):
        self.buffer = deque(maxlen=size)
        self.maxSize = size
        self.len = 0
        
        self.path_logs = "/src/shared/logs"
        if not os.path.isdir(self.path_logs):
            os.mkdir(self.path_logs)

    def sample(self, count):
        """
        samples a random batch from the replay memory buffer
        :param count: batch size
        :return: batch (numpy array)
        """
        count = min(count, self.len)
        batch = random.sample(self.buffer, count)

        s_arr = np.float32([arr[0] for arr in batch])
        a_arr = np.float32([arr[1] for arr in batch])
        r_arr = np.float32([arr[2] for arr in batch])
        s1_arr = np.float32([arr[3] for arr in batch])

        return s_arr, a_arr, r_arr, s1_arr

    def len(self):
        return self.len
        
    def log(self):
        df = pd.DataFrame(self.buffer, columns=['state', 'action', 'reward', 'new_state', 'episode_id'])
        now = datetime.now()
        filename = '/'+str(now.year)+'_'+str(now.month)+'_'+str(now.day)+'_'+str(now.hour)+'_'+str(now.minute)+'_'+str(now.second)
        df.to_csv(self.path_logs+filename+".csv")

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
        if self.len > self.maxSize:
            self.len = self.maxSize
        else:
            self.len += 1
        self.buffer.append(transition)

